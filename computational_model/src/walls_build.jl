
#In this script, we instantiate the RL environment which include initialize() and step() functions.

using Flux, Statistics, Random, Distributions, StatsFuns, Zygote, PyPlot, Logging

"""function that computes things like the input and output dimensionality of the network"""
function useful_dimensions(Larena, planner)
    Nstates = Larena^2 #number of states in arena
    Nstate_rep = 2 #dimensionality of the state representation (e.g. '2' for x,y-coordinates)
    Naction = 5 #number of actions available
    Nout = Naction + 1 + Nstates #actions and value function and prediction of state
    Nwall_in = 2 * Nstates #provide full info
    Nin = Naction + 1 + 1 + Nstates + Nwall_in #5 actions, 1 rew, 1 time, L^2 states, some walls

    Nin += planner.Nplan_in #additional inputs from planning
    Nout += planner.Nplan_out #additional outputs for planning

    return Nstates, Nstate_rep, Naction, Nout, Nin
end

"""function that objects the position of the agent given an action (assuming no walls)"""
function update_agent_state(agent_state, amove, Larena)
    new_agent_state =
        agent_state + [amove[1:1, :] - amove[2:2, :]; amove[3:3, :] - amove[4:4, :]] #2xbatch
    new_agent_state = Int32.((new_agent_state .+ Larena .- 1) .% Larena .+ 1) #1:L (2xbatch)
    return new_agent_state
end

"""
    act_and_receive_reward(action, world_state, planning, env_dimensions, agent_output, model, hidden_state, model_properties)
output: reward, new_world_state, ground_truth_predictions, one-hot-action, agent_at_reward?
This function implements the 'environment' of the RL algorithm.
It takes as input the output of the agent and the state of the world and returns the new state of the world.
Note that 'planning' in our formulation takes place in the environment, so the output includes the result of planning.
"""
function act_and_receive_reward(
    a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, mp
)
    agent_state = world_state.agent_state
    environment_state = world_state.environment_state
    reward_location = environment_state.reward_location
    wall_loc = environment_state.wall_loc
    Naction = environment_dimensions.Naction
    Larena = environment_dimensions.Larena

    agent_state_ind = state_ind_from_state(Larena, agent_state) #extract index
    batch = size(a, 2) #batch size
    Nstates = Larena^2

    ahot = zeros(Naction, batch) #attempted action
    amove = zeros(Naction, batch) #actual movement
    rew = zeros(Float32, 1, batch) #reward collected

    #construct array of attempted and actual movements
    for b in 1:batch
        abatch = a[1, b] # action
        ahot[abatch, b] = 1f0 #attempted action
        if (abatch < 4.5) && Bool(wall_loc[agent_state_ind[b], abatch, b])
            rew[1, b] -= 0f0 #penalty for hitting wall?
        else
            amove[abatch, b] = 1 #only move if we don't hit a wall
        end
    end

    new_agent_state = update_agent_state(agent_state, amove, Larena) #(x,y) coordinates
    shot = onehot_from_state(Larena, new_agent_state) #one-hot encoding (Nstates x batch)
    s_index = reduce(vcat, [sortperm(-shot[:, b])[1] for b in 1:batch]) #corresponding index
    r_index = get_rew_locs(reward_location) #index of reward location
    predictions = (Int32.(s_index), Int32.(r_index)) #things to be predicted by the agent

    found_rew = Bool.(reward_location[Bool.(shot)]) #moved to the reward
    s_old_hot = onehot_from_state(Larena, agent_state) #one-hot encoding of previous agent_state
    at_rew = Bool.(reward_location[Bool.(s_old_hot)]) #at reward before action

    moved = sum(amove[1:4, :]; dims=1)[:] #did I perform a movement? (size batch)
    rew[1, found_rew .& (moved .> 0.5)] .= 1 #get reward if agent moved to reward location

    ### teleport the agents that found the reward on the previous iteration ###
    for b in 1:batch
        if at_rew[b] #at reward
            tele_reward_location = ones(Nstates) / (Nstates - 1) #where can I teleport to (not rew location)
            tele_reward_location[Bool.(reward_location[:, b])] .= 0
            new_state = rand(Categorical(tele_reward_location), 1, 1) #sample new state uniformly at random
            new_agent_state[:, b] = state_from_loc(Larena, new_state) #convert to (x,y) coordinates
            shot[:, b] .= 0f0; shot[new_state[1], b] = 1f0 #update onehot location
        end
    end

    #run planning algorithm
    planning_state, plan_inds = planner.planning_algorithm(world_state,
                                                            ahot,
                                                            environment_dimensions,
                                                            agent_output,
                                                            at_rew,
                                                            planner,
                                                            model,
                                                            h_rnn,
                                                            mp)

    planned = Bool.(zeros(batch)); planned[plan_inds] .= true #which agents within the batch engaged in planning

    #update the time elapsed for each episode
    new_time = copy(environment_state.time)
    new_time[ .~ planned ] .+= 1f0 #increment time for acting
    if planner.constant_rollout_time
        new_time[planned] .+= planner.planning_time #increment time for planning
    else
        plan_states = planning_state.plan_cache
        plan_lengths = sum(plan_states[:, planned] .> 0.5, dims = 1)[:] # number of planning steps for each batch
        new_time[planned] += plan_lengths*planner.planning_time/5
        println("variabled planning time! ", plan_lengths*planner.planning_time/5)
    end

    rew[1, planned] .+= planner.planning_cost #cost of planning (in units of rewards; default 0)

    #update the state of the world
    new_world_state = WorldState(;
        agent_state=new_agent_state,
        environment_state=WallState(; wall_loc, reward_location, time = new_time),
        planning_state=planning_state
    )

    return Float32.(rew), new_world_state, predictions, ahot, at_rew
end

"""
    build_environment(arena_size, N_hidden, max_time, planning_depth, greedy_actions, no_planning)
This function constructs an environment object which includes 'initialize' and 'step' methods for the agent to interact with.
"""
function build_environment(
    Larena::Int,
    Nhidden::Int,
    T::Int;
    Lplan::Int,
    greedy_actions=false,
    no_planning = false,
    constant_rollout_time = true,
)

    #create planner object
    #note that planner includes a 'plan_state' which can carry over in more general planning algorithms
    planner, initial_plan_state = build_planner(Lplan, Larena; constant_rollout_time)
    Nstates, Nstate_rep, Naction, Nout, Nin = useful_dimensions(Larena, planner) #compute some useful quantities
    model_properties = ModelProperties(Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning) #initialize a model property object
    environment_dimensions = EnvironmentDimensions(Nstates, Nstate_rep, Naction, T, Larena) #initialize an environment dimension object

    ### define a 'step' function that updates the environment ###
    function step(agent_output, a, world_state, environment_dimensions, model_properties, model, h_rnn)

        Zygote.ignore() do #no differentiation through the environment
            rew, new_world_state, predictions, ahot, at_rew = act_and_receive_reward(
                a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, model_properties
            ) #take a step through the environment
            #generate agent input
            agent_input = gen_input(new_world_state, ahot, rew, environment_dimensions, model_properties)
            #return reward, input, world state and ground truths for predictions
            return rew, Float32.(agent_input), new_world_state, predictions
        end
    end

    #create initialization function
    function initialize(reward_location, agent_state, batch, mp; initial_params = [])
        return initialize_arena(reward_location, agent_state, batch, mp, environment_dimensions, initial_plan_state, initial_params=initial_params)
    end

    #construct environment with initialize() and step() functions and a list of dimensions
    environment = Environment(initialize, step, environment_dimensions)

    ### task specific evaluation/progress function ####
    function model_eval(m, batch::Int64, loss_hp::LossHyperparameters)
        Nrep = 5
        means = zeros(Nrep, batch)
        all_actions = zeros(Nrep, batch)
        firstrews = zeros(Nrep, batch)
        preds = zeros(T - 1, Nrep, batch)
        Naction = environment_dimensions.Naction
        Nstates = environment_dimensions.Nstates

        for i in 1:Nrep
            _, agent_outputs, rews, actions, world_states, _ = run_episode(
                m, environment, loss_hp; hidden=true, batch=batch
            )
            agent_states = mapreduce(
                (x) -> x.agent_state, (x, y) -> cat(x, y; dims=3), world_states
            )
            means[i, :] = sum(rews .>= 0.5; dims=2) #compute total reward for each batch
            all_actions[i, :] = (mean(actions .== 5; dims=2) ./ mean(actions .> 0.5, dims = 2)) #fraction of standing still for each batch
            for b in 1:batch
                firstrews[i, b] = sortperm(-(rews[b, :] .> 0.5))[1] #time to first reward
            end
            for t in 1:(T - 1)
                for b in 1:batch
                    pred = sortperm(
                        -agent_outputs[(Naction + 1 + 1):(Naction + 1 + Nstates), b, t]
                    )[1] #predicted agent_state at time t (i.e. prediction of agent_state(t+1))
                    agent_state = Int32.(agent_states[:, b, t + 1]) #true agent_state(t+1)
                    preds[t, i, b] = onehot_from_state(Larena, agent_state)[Int32(pred)] #did we get it right?
                end
            end

        end
        return mean(means),
        mean(preds), mean(all_actions), mean(firstrews)
    end

    return model_properties, environment, model_eval
end
