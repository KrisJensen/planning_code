#in this script, we evaluate the performance of our original model
#and compare this to a model where the rollout times have been shuffled
#to see whether the structured rollout timings of the agent are important for performance

# load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan

epoch = plan_epoch #model training epoch to use for evaluation (default to final epoch)
results = Dict() #container for storing results
batch = 50000 #number of episodes to simulate

for seed = seeds #for each independently trained model

    plan_ts, Nact, Nplan = [], [], [] #containers for storing data
    results[seed] = Dict() #dict for this model
    for shuffle = [false; true] #run both the non-shuffled and shuffled replays
        Random.seed!(1) #set random seed for identical arenas across the two scenarios

        # load model parameters and create environment
        network, opt, store, hps, policy, prediction = recover_model("$(loaddir)N100_T50_Lplan8_seed$(seed)_$epoch")
        model_properties, environment, model_eval = build_environment(
            hps["Larena"], hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions, no_planning = shuffle)
        m = ModularModel(model_properties, network, policy, prediction, forward_modular) #construct model

        #extract some useful parameters
        ed = environment.dimensions
        Nhidden = m.model_properties.Nhidden
        T = ed.T

        #initialize environment
        world_state, agent_input = environment.initialize(zeros(2), zeros(2), batch, m.model_properties, initial_params = [])
        agent_state = world_state.agent_state
        h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)); #expand hidden state

        rews, as = [], []
        rew = zeros(batch)
        iter = 0
        while any(world_state.environment_state.time .< (T+1 - 1e-2)) #run until completion
            iter += 1 #count iteration
            h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step
            a[rew .> 0.5] .= 1f0 #no 'planning' at reward

            if shuffle #if we're shuffling the replay times
                for b = 1:batch #for each episode
                    if iter in plan_ts[b] #this is a shuffled time
                        if rew[b] < 0.5 #if we're not at the reward
                            a[b] = 5f0 #perform a rollout
                        else #if we're at the reward location, resample a new rollout iteration
                            remaining = Set(iter+1:Nact[b]-3) #set of remaining iteration
                            options = setdiff(remaining, plan_ts[b]) #consider the ones where we are not already planning to do a rollout
                            if length(options) > 0 #if there are other iterations left
                                push!(plan_ts[b], rand(options)) #sample a new rollout iteration
                            end
                        end
                    end
                end
            end

            active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes
            #take an environment step
            rew, agent_input, world_state, predictions = environment.step(
                agent_output, a, world_state, environment.dimensions, m.model_properties,
                m, h_rnn
            )
            rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #mask episodes that are finished
            push!(rews, rew); push!(as, a) #store rewards and actions from this iteration
        end

        rews, as = [reduce(vcat, arr) for arr = [rews, as]] #combine rewards and actions into array
        Nact, Nplan = sum(as .> 0.5, dims = 1), sum(as .> 4.5, dims = 1) #number of actions and number of plans in each episode
        plan_ts = [Set(randperm(Nact[b]-3)[1:Nplan[b]]) for b = 1:batch] #resample the iterations at which I should plan (avoiding last iterations)

        #print some summary data
        println(seed, " ", shuffle)
        println("reward: ", sum(rews .> 0.5) / batch) #reward
        println("planning fraction: ", sum(as .> 4.5) / sum(as .> 0.5)) #planning fraction
        results[seed][shuffle] = rews #store the rewards for this experiment
        
    end
end

#store result
@save "$(datadir)/performance_shuffled_planning.bson" results
