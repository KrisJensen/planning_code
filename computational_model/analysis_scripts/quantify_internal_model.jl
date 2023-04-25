# in this script, we quantify the accuracy of the internal world model over training time

# load scripts
include("anal_utils.jl")
using ToPlanOrNotToPlan, NaNStatistics

batch = 1000 #number of environments to consider
results = Dict() #dictionary for storing results

for seed = seeds #for each independently trained RL agent
    results[seed] = Dict() #results for this model
    for epoch = 0:50:1000 #for each training epoch

        # seed random seed for reproducibility
        Random.seed!(1)

        filename = "N100_T50_Lplan8_seed$(seed)_$epoch" #model to load
        network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters

        #initialize environment and model
        Larena = hps["Larena"]
        model_properties, environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        #extract some useful parameters
        ed = environment.dimensions
        Nout = m.model_properties.Nout
        Nhidden = m.model_properties.Nhidden
        T, Naction, Nstates = ed.T, ed.Naction, ed.Nstates

        ### initialize reward probabilities and state ###
        world_state, agent_input = environment.initialize(zeros(2), zeros(2), batch, m.model_properties, initial_params = [])
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state

        #containers for storing prediction results
        rew_preds, state_preds = zeros(batch, 200) .+ NaN, zeros(batch, 200) .+ NaN
        exploit = Bool.(zeros(batch)) #are we in the exploitation phase
        iter = 1 #iteration number
        rew, old_rew = zeros(batch), zeros(batch) #containers for storing reward information

        #iterate through RL agent/environment
        while any(world_state.environment_state.time .< (T+1 - 1e-2))
            iter += 1 #update iteration number
            h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step
            active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes

            old_rew[:] = rew[:] #did I get reward on previous timestep?
            #update environment given action and current state
            rew, agent_input, world_state, predictions = environment.step(
                agent_output, a, world_state, environment.dimensions, m.model_properties,
                m, h_rnn
            )

            #extract true next state and reward location
            strue = [coord[1] for coord = argmax(onehot_from_state(Larena, world_state.agent_state), dims = 1)][:]
            rtrue = [coord[1] for coord = argmax(world_state.environment_state.reward_location, dims = 1)][:]

            #calculate reward prediction accuracy
            i1, i2 = (Naction + Nstates + 2), (Naction + Nstates + 1 + Nstates) #indices of corresponding output
            rpred = [coord[1] for coord = argmax(agent_output[i1:i2, :], dims = 1)][:] #extract prediction output
            inds = findall(exploit .& active) #only consider exploitation
            rew_preds[inds, iter] = Float64.(rpred .== rtrue)[inds] #store binary 'success' data

            ### calculate state accuracy ###
            i1, i2 = (Naction + 1 + 1), (Naction + 1 + Nstates) #indices of corresponding output
            spred = [coord[1] for coord = argmax(agent_output[i1:i2, :], dims = 1)][:] #extract prediction output
            inds = findall((old_rew .< 0.5) .& active) #ignore teleportation step
            state_preds[inds, iter] = Float64.(spred .== strue)[inds] #store binary 'success' data

            exploit[old_rew .> 0.5] .= true #indicate which episodes are in the exploitation phase
            rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #mask if episode is finished
        end

        println(seed, " ", epoch, " ", nanmean(rew_preds), " ", nanmean(state_preds))
        results[seed][epoch] = Dict("rew" => nanmean(rew_preds), "state" => nanmean(state_preds)) #store result
    end
end

@save "$(datadir)/internal_model_accuracy.bson" results

