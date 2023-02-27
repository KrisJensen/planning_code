using Flux, Statistics, Random, Distributions, StatsFuns, Zygote, PyPlot, Logging

function useful_dimensions(Larena, planner)
    Nstates = Larena^2
    Nstate_rep = 2
    Naction = 5
    Nout = Naction + 1 + Nstates #actions and value function and prediction of state + reward
    Nwall_in = 2 * Nstates #provide full info
    Nin = Naction + 1 + 1 + Nstates + Nwall_in #5 actions, 1 rew, 1 time, L^2 states, some walls

    Nin += planner.Nplan_in
    Nout += planner.Nplan_out

    return Nstates, Nstate_rep, Naction, Nout, Nin
end

function update_agent_state(agent_state, amove, Larena)
    new_agent_state =
        agent_state + [amove[1:1, :] - amove[2:2, :]; amove[3:3, :] - amove[4:4, :]] #2xbatch
    new_agent_state = Int32.((new_agent_state .+ Larena .- 1) .% Larena .+ 1) #1:L (2xbatch)
    return new_agent_state
end

function act_and_receive_reward(
    a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, mp
)
    agent_state = world_state.agent_state
    environment_state = world_state.environment_state
    reward_location = environment_state.reward_location
    wall_loc = environment_state.wall_loc
    Naction = environment_dimensions.Naction
    Larena = environment_dimensions.Larena

    agent_state_ind = state_ind_from_state(Larena, agent_state) #Batch
    batch = size(a, 2)
    Nstates = Larena^2

    ahot = zeros(Naction, batch) #attempted action
    amove = zeros(Naction, batch) #actual movement

    rew = zeros(Float32, 1, batch)

    for b in 1:batch
        abatch = a[1, b] # action
        ahot[abatch, b] = 1f0 #attempted action
        if (abatch < 4.5) && Bool(wall_loc[agent_state_ind[b], abatch, b])
            rew[1, b] -= 0f0 #penalty for hitting wall?
        else
            amove[abatch, b] = 1 #only move if we don't hit a wall
        end
    end

    new_agent_state = update_agent_state(agent_state, amove, Larena)
    shot = onehot_from_state(Larena, new_agent_state) #one-hot encoding (Nstates x batch)

    #'s_index' the index corresponding to the agent_state, and 'wall_loc' is the location of the walls
    s_index = reduce(vcat, [sortperm(-shot[:, b])[1] for b in 1:batch])
    r_index = get_rew_locs(reward_location)
    predictions = (Int32.(s_index), Int32.(r_index))

    found_rew = Bool.(reward_location[Bool.(shot)]) #found reward
    s_old_hot = onehot_from_state(Larena, agent_state) #one-hot encoding of previous agent_state
    at_rew = Bool.(reward_location[Bool.(s_old_hot)]) #at reward before action

    moved = sum(amove[1:4, :]; dims=1)[:] #did I perform a movement? (size batch)
    rew[1, found_rew .& (moved .> 0.5)] .= 1

    for b in 1:batch
        if at_rew[b] #at reward
            tele_reward_location = ones(Nstates) / (Nstates - 1) #where can I teleport to (not rew location)
            tele_reward_location[Bool.(reward_location[:, b])] .= 0
            new_state = rand(Categorical(tele_reward_location), 1, 1)
            new_agent_state[:, b] = state_from_loc(Larena, new_state)
            shot[:, b] .= 0f0; shot[new_state[1], b] = 1f0 #update onehot location
        end
    end

    planning_state, plan_inds = planner.planning_algorithm(world_state,
                                                            ahot,
                                                            environment_dimensions,
                                                            agent_output,
                                                            at_rew,
                                                            planner,#)
                                                            model,
                                                            h_rnn,
                                                            mp)

    planned = Bool.(zeros(batch)); planned[plan_inds] .= true #where did we plan?
    new_time = copy(environment_state.time)
    new_time[ .~ planned ] .+= 1f0 #increment time for acting
    new_time[planned] .+= planner.planning_time #increment time for planning

    rew[1, planned] .+= planner.planning_cost #cost of planning (in rewards; default 0)

    new_world_state = WorldState(;
        agent_state=new_agent_state,
        environment_state=WallState(; wall_loc, reward_location, time = new_time),
        planning_state=planning_state
    )

    return Float32.(rew), new_world_state, predictions, ahot, at_rew
end

function build_environment(
    Larena::Int,
    Nhidden::Int,
    T::Int;
    Lplan::Int,
    greedy_actions=false,
    no_planning = false,
)

    planner, initial_plan_state = build_planner(Lplan, Larena)
    Nstates, Nstate_rep, Naction, Nout, Nin = useful_dimensions(Larena, planner)
    model_properties = ModelProperties(Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning)
    environment_dimensions = EnvironmentDimensions(Nstates, Nstate_rep, Naction, T, Larena)

    ### specify environment function per task ###
    function step(agent_output, a, world_state, environment_dimensions, model_properties, model, h_rnn)

        Zygote.ignore() do
            rew, new_world_state, predictions, ahot, at_rew = act_and_receive_reward(
                a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, model_properties
            )

            agent_input = gen_input(new_world_state, ahot, rew, environment_dimensions, model_properties)

            return rew, Float32.(agent_input), new_world_state, predictions
        end
    end

    function initialize(reward_location, agent_state, batch, mp; initial_params = [])
        return initialize_arena(reward_location, agent_state, batch, mp, environment_dimensions, initial_plan_state, initial_params=initial_params)
    end

    environment = Environment(initialize, step, environment_dimensions)

    ### task specific evaluation/progress function ####
    function model_eval(m, batch::Int64, loss_hp::LossHyperparameters)
        #evaluation function; compute mean reward (random is ~T/Nstates*0.8; oracle is (T-Nstates/2)/3)
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
