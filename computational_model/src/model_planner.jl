using Distributions

function model_tree_search(goal, world_state, model, h_rnn, plan_inds, times, ed, mp, planner; Print = false)

    Larena, Naction = ed.Larena, ed.Naction
    Nstates = Larena^2

    batch = size(h_rnn, 2)
    path = zeros(4, planner.Lplan, batch)
    all_Vs = zeros(Float32, batch) #value functions
    found_rew = zeros(Float32, batch) #did I finish planning
    plan_states = zeros(Int32, planner.Lplan, batch)
    wall_loc = world_state.environment_state.wall_loc

    #only consider planning states
    h_rnn = h_rnn[:, plan_inds]
    goal = goal[plan_inds]
    times = times[plan_inds]
    wall_loc = wall_loc[:, :, plan_inds]
    ytemp = h_rnn #same for GRU

    agent_input = zeros(Float32, mp.Nin) #instantiate
    new_world_state = world_state

    for n_steps = 1:planner.Lplan
        batch = length(goal) #number of active states

        if n_steps > 1.5 #start from current hidden state
            ### generate new output ###
            h_rnn, ytemp = model.network[GRUind].cell(h_rnn, agent_input) #forward pass
        end

        ### generate actions from hidden activity ###
        logπ_V = model.policy(ytemp)
        #normalize over actions
        logπ = logπ_V[1:4, :] .- Flux.logsumexp(logπ_V[1:4, :], dims = 1) #softmax
        Vs = logπ_V[6, :] / 10f0 #range ~ [0,1]

        πt = exp.(logπ)
        a = zeros(Int32, 1, batch) #sample actions
        a[:] = Int32.(rand.(Categorical.([πt[:, b] for b = 1:batch])))
        #a[:] = Int32.(rand.(Categorical.([ones(4) / 4 for b = 1:batch]))) #random action

        ### record actions ###
        for (ib, b) = enumerate(plan_inds)
            path[a[1, ib], n_steps, b] = 1f0 #'a' in local coordinates, 'path' in global
        end

        ### generate predictions ###
        ahot = zeros(Float32, Naction, batch) #one-hot
        for b = 1:batch ahot[a[1, b], b] = 1f0 end
        prediction_input = [ytemp; ahot] #input to prediction module
        prediction_output = model.prediction(prediction_input) #output from prediction module

        ### draw new states ###
        spred = prediction_output[1:Nstates, :] #predicted states (Nstates x batch)
        spred = spred .- Flux.logsumexp(spred; dims=1) #softmax over states
        state_dist = exp.(spred) #state distribution
        new_states = Int32.(argmax.([state_dist[:, b] for b = 1:batch])) #maximum likelihood new states

        Print && println(n_steps, " ", batch, " ", mean(maximum(πt, dims = 1)), " ", mean(maximum(state_dist, dims = 1)))

        ### record information about having finished ###
        not_finished = findall(new_states .!= goal) #vector of states that have not finished!
        finished = findall(new_states .== goal) #found the goal location on these ones

        all_Vs[plan_inds] = Vs #store latest value
        plan_states[n_steps, plan_inds] = new_states #store states
        found_rew[plan_inds[finished]] .+= 1f0 #record where we found the goal location

        if length(not_finished) == 0 return path, all_Vs, found_rew, plan_states end #finish if all done

        ### only consider active states going forward ###
        h_rnn = h_rnn[:, not_finished]
        goal = goal[not_finished]
        plan_inds = plan_inds[not_finished]
        times = times[not_finished] .+ 1f0 #increment time
        wall_loc = wall_loc[:, :, not_finished] 
        reward_location = onehot_from_loc(Larena, goal) #onehot
        xplan = zeros(Float32, planner.Nplan_in, length(goal)) #no planning input

        ###reward inputs ###
        rew = zeros(Float32, 1, length(not_finished)) #we continue with the ones that did not get reward

        ### update world state ###
        new_world_state = WorldState(;
                    agent_state=state_from_loc(Larena, new_states[not_finished]'),
                    environment_state=WallState(; wall_loc, reward_location, time = times),
                    planning_state = PlanState(xplan, nothing)
                )

        ### generate input ###
        agent_input = gen_input(new_world_state, ahot[:, not_finished], rew, ed, mp)

    end
    return path, all_Vs, found_rew, plan_states
end

function model_planner(world_state,
    ahot,
    ed,
    agent_output,
    at_rew,
    planner,
    model,
    h_rnn,
    mp;
    Print = false,
    returnall = false,
    true_transition = false
)

    Larena = ed.Larena
    Naction = ed.Naction
    Nstates = ed.Nstates
    batch = size(ahot, 2)
    times = world_state.environment_state.time

    plan_inds = findall(Bool.(ahot[5, :]) .& (.~at_rew)) #everywhere we stand still not at the reward
    rpred = agent_output[(Naction + Nstates + 2):(Naction + Nstates + 1 + Nstates), :]
    goal = [argmax(rpred[:, b]) for b = 1:batch] #index of ML goal location

    ### agent-driven planning ###
    path, all_Vs, found_rew, plan_states = model_tree_search(goal, world_state, model, h_rnn, plan_inds, times, ed, mp, planner, Print = Print)

    xplan = zeros(planner.Nplan_in, batch)
    for b = 1:batch
        xplan[:, b] = [path[:, :, b][:]; found_rew[b]]
    end
    planning_state = PlanState(xplan, plan_states)

    returnall && return planning_state, plan_inds, (path, all_Vs, found_rew, plan_states)
    return planning_state, plan_inds
end
