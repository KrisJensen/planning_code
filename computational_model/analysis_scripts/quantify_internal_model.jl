## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics
using BSON: @save

loss_hp = LossHyperparameters(0, 0, 0, 0, 0, 0, 1000, true, 0f0, () -> ())
Save = true

greedy_actions = true
no_planning = false
seeds = 61:65
epoch = plan_epoch
batch = 1000

results = Dict()

for seed = seeds
    results[seed] = Dict()
    for epoch = 0:100:2000
        network, opt, store, hps, policy, prediction = recover_model("../models/maze/N100_T50_seed$(seed)_Lplan8_$epoch", modular = true)

        Larena = hps["Larena"]
        model_properties, environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        Nstates = Larena^2
        Naction = environment.properties.dimensions.Naction

        ##
        Random.seed!(1)

        ed = environment.properties.dimensions
        Nout = m.model_properties.Nout
        Nhidden = m.model_properties.Nhidden
        T = ed.T

        ### initialize reward probabilities and state ###
        world_state, agent_input = environment.initialize(zeros(2), zeros(2), batch, m.model_properties, initial_params = [])
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state

        rew_preds, state_preds = zeros(batch, 200) .+ NaN, zeros(batch, 200) .+ NaN
        exploit = Bool.(zeros(batch))
        iter = 1
        rew, old_rew = zeros(batch), zeros(batch)
        while any(world_state.environment_state.time .< (T+1 - 1e-2))
            iter += 1
            h_rnn, agent_output, a = m.forward(m, ed, agent_input; h_rnn=h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)
            active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active heads

            old_rew[:] = rew[:] #did I get reward on previous timestep
            rew, agent_input, world_state, predictions = environment.step(
                agent_output, a, world_state, environment.properties, m.model_properties,
                m, h_rnn
            )

            strue = [coord[1] for coord = argmax(onehot_from_state(Larena, world_state.agent_state), dims = 1)][:]
            rtrue = [coord[1] for coord = argmax(world_state.environment_state.reward_location, dims = 1)][:]

            ### calculate reward accuracy ###
            i1, i2 = (Naction + Nstates + 2), (Naction + Nstates + 1 + Nstates)
            rpred = [coord[1] for coord = argmax(agent_output[i1:i2, :], dims = 1)][:]
            inds = findall(exploit .& active)
            rew_preds[inds, iter] = Float64.(rpred .== rtrue)[inds]

            ### calculate state accuracy ###
            i1, i2 = (Naction + 1 + 1), (Naction + 1 + Nstates)
            spred = [coord[1] for coord = argmax(agent_output[i1:i2, :], dims = 1)][:]
            inds = findall((old_rew .< 0.5) .& active) #ignore teleportation step
            state_preds[inds, iter] = Float64.(spred .== strue)[inds]

            exploit[old_rew .> 0.5] .= true
            rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #mask if episode is finished
        end

        println(seed, " ", epoch, " ", nanmean(rew_preds), " ", nanmean(state_preds))
        #println(sum(.~isnan.(rew_preds))/batch, " ", sum(.~isnan.(state_preds))/batch)

        results[seed][epoch] = Dict("rew" => nanmean(rew_preds), "state" => nanmean(state_preds))
    end
end

@save "$(datadir)/internal_model_accuracy.bson" results

