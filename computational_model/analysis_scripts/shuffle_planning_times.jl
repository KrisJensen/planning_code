## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics
using BSON: @save

loss_hp = LossHyperparameters(0, 0, 0, 0, 0, 0, 1000, true, 0f0, () -> ())

greedy_actions = true
seeds = 61:65
epoch = plan_epoch
results = Dict()
batch = 50000

for seed = seeds
    ##
    plan_ts, Nact, Nplan = [], [], []
    results[seed] = Dict()
    for shuffle = [false; true]
        Random.seed!(1)

        network, opt, store, hps, policy, prediction = recover_model("../models/maze/N100_T50_seed$(seed)_Lplan8_$epoch", modular = true)
        model_properties, wall_environment, model_eval = build_environment(
            hps["Larena"], hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions, no_planning = shuffle)
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)
        Naction = wall_environment.properties.dimensions.Naction
        environment = wall_environment

        tic = time()
        ed = environment.properties.dimensions
        Nout = m.model_properties.Nout
        Nhidden = m.model_properties.Nhidden
        T = ed.T

        ### initialize reward probabilities and state ###
        world_state, agent_input = environment.initialize(zeros(2), zeros(2), batch, m.model_properties, initial_params = [])
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)); #expand hidden state
        rews, as = [], []
        rew = zeros(batch)
        iter = 0
        while any(world_state.environment_state.time .< (T+1 - 1e-2))
            iter += 1
            h_rnn, agent_output, a = m.forward(m, ed, agent_input; h_rnn=h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)
            a[rew .> 0.5] .= 1f0 #no 'planning' at reward
            if shuffle
                for b = 1:batch
                    if iter in plan_ts[b] #need to plan
                        if rew[b] > 0.5
                            #nothing
                            remaining = Set(iter+1:Nact[b]-3)
                            options = setdiff(remaining, plan_ts[b])
                            if length(options) > 0
                                push!(plan_ts[b], rand(options))
                            end
                        else
                            a[b] = 5f0
                        end 
                    end
                end
            end

            active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active heads
            rew, agent_input, world_state, predictions = environment.step(
                agent_output, a, world_state, environment.properties, m.model_properties,
                m, h_rnn
            )
            rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #mask if episode is finished
            push!(rews, rew); push!(as, a)
        end
        rews, as = [reduce(vcat, arr) for arr = [rews, as]]
        if shuffle
            match = findall(sum(as .> 4.5, dims = 1)[:] .== Nplan[:])
            results[seed]["match"] = match
            println(length(match)/batch)
        end
        
        Nact, Nplan = sum(as .> 0.5, dims = 1), sum(as .> 4.5, dims = 1)
        plan_ts = [Set(randperm(Nact[b]-3)[1:Nplan[b]]) for b = 1:batch] #when should I plan?

        Tmax = size(as, 2)
        println(seed, " ", shuffle)
        println(sum(rews .> 0.5) / batch, " ", time() - tic)
        println("planning fraction: ", sum(as .> 4.5) / sum(as .> 0.5))
        if shuffle println(sum(rews[:,match] .> 0.5)/length(match)) end
        results[seed][shuffle] = rews
        
    end
end

@save "$(datadir)/performance_shuffled_planning.bson" results


using BSON: @load
@load "$(datadir)/performance_shuffled_planning.bson" results
ress = zeros(length(seeds), 2)
for (i, shuffle) = enumerate([true; false])
    for (iseed, seed) = enumerate(seeds)
        rews = results[seed][shuffle]
        keep = results[seed]["match"]
        ress[iseed, i] = sum(rews) / size(rews, 2)
        #ress[iseed, i] = sum(rews[:,keep] .> 0.5)/length(keep)
    end
end
m, s = mean(ress, dims = 1)[:], std(ress, dims = 1)[:]/sqrt(length(seeds))

