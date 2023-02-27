function random_policy(x, md, ed; stay=true)
    #if stay is false, only uniform over actual actions
    batch = size(x, 2)
    ys = Float32.(zeros(md.Nout, batch))
    if stay
        ys[1:(ed.Naction), :] .= log(1 / ed.Naction)
    else
        ys[1:(ed.Naction - 1), :] .= log(1 / (ed.Naction - 1))
        ys[ed.Naction, :] .= -Inf
    end
    return ys
end

function dist_to_rew(ps, wall_loc, Larena; fname="figs/wall/min_dist.png")
    #compute geodesic distance to reward from each state (i.e. taking walls into account)
    #ps is Nstates x 1
    #wall_loc is 16x4x1
    Nstates = Larena^2
    deltas = [[1; 0], [-1; 0], [0; 1], [0; -1]] #transitions for each action
    rew_loc = state_from_onehot(Larena, ps) #2x1
    dists = zeros(Larena, Larena) .+ NaN #distances to goal
    dists[rew_loc[1], rew_loc[2]] = 0 #reward has zero distance
    live_states = Bool.(zeros(Nstates))
    live_states[state_ind_from_state(Larena, rew_loc)[1]] = true #start from rew loc and work backwards
    for step in 1:(Nstates - 1) #steps from reward
        for state_ind in findall(live_states) #all states I was at in (step-1) steps
            state = state_from_loc(Larena, state_ind)
            for a in 1:4 #for each action
                if ~Bool(wall_loc[state_ind, a, 1]) #if I do not hit a wall
                    newstate = state .+ deltas[a] #where do I end up in 'step' steps
                    newstate = Int.((newstate .+ Larena .- 1) .% Larena .+ 1) #1:L (2xbatch)
                    if isnan(dists[newstate[1], newstate[2]]) #if I haven't gotten here in fewer steps
                        dists[newstate[1], newstate[2]] = step #got here in step steps
                        new_ind = state_ind_from_state(Larena, newstate)[1]
                        live_states[new_ind] = true #need to search from here for >step steps
                    end
                end
            end
            live_states[state_ind] = false #done searching for this state
        end
    end
    return dists #return geodesics
end

function optimal_policy(state, wall_loc, dists, ed)
    #return uniform log policy over actions that minimize the path length to goal
    #state is 2x1
    #wall_loc is Nstates x 4 x 1
    #dists is Larena x Larena of geodesic distance (from dist_to_rew())
    Naction, Larena = ed.Naction, ed.Larena
    deltas = [[1; 0], [-1; 0], [0; 1], [0; -1]] #transitions for each action
    nn_dists = zeros(4) .+ Inf #distance to reward for each action
    state_ind = state_ind_from_state(Larena, state)[1] #where am I
    for a in 1:4 #for each action
        if ~Bool(wall_loc[state_ind, a, 1]) #if I do not hit a wall
            newstate = state .+ deltas[a] #where am I now
            newstate = Int.((newstate .+ Larena .- 1) .% Larena .+ 1) #1:L (2xbatch)
            nn_dists[a] = dists[newstate[1], newstate[2]] #how far is this from reward
        end
    end
    as = findall(nn_dists .== minimum(nn_dists)) #all optimal actions
    πt = zeros(Naction)
    πt[as] .= 1 / length(as) #uniform policy
    return πt #optimal policy
end