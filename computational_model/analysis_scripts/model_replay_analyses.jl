#in this script, we repeat the analyses performed for the hippocampal replay data

#load some libraries
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics

epoch = plan_epoch #training epoch to evaluate
res_dict = Dict() #container for storing results

for seed = seeds #iterate through independently trained RL agents

println("\n new seed $(seed)!")
res_dict[seed] = Dict() #results for this agent

filename = "N100_T50_Lplan8_seed$(seed)_$epoch" #model to load
network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters

#construct RL environment
Larena = hps["Larena"]
model_properties, wall_environment, model_eval = build_environment(
    Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
)
#construct RL agent
m = ModularModel(model_properties, network, policy, prediction, forward_modular)

#extract some useful parameters
ed = wall_environment.dimensions
Nout, Nhidden = m.model_properties.Nout, m.model_properties.Nhidden
Nstates, Naction, T = ed.Nstates,  ed.Naction, ed.T

Random.seed!(2) #set seed for reproducibility
batch = 25000 #number of environments to consider

#initialize environment
println("simulating agent")
world_state, agent_input = wall_environment.initialize(
    zeros(2), zeros(2), batch, m.model_properties
)
agent_state = world_state.agent_state
h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
rew = zeros(batch) #container for storing reward info

# initialize containers for storing data
tmax = 200 #number of iterations to run
exploit = Bool.(zeros(batch)) #are we in the exploitation phase
Lplan = model_properties.Lplan #planning depth
plans = zeros(batch, tmax, Lplan) .+ NaN; #container for saving rollouts
plan_as = zeros(batch, tmax, Lplan) .+ NaN; #container for saving rollout actions
planner, initial_plan_state = build_planner(Lplan, Larena) #initialize planning module
agent_states = zeros(batch, 2, tmax);
actions = zeros(batch, tmax)
all_rews = []

rewlocs = [argmax(world_state.environment_state.reward_location[:, i]) for i = 1:batch] #reward locations
success, success_cv = [zeros(batch, tmax, 16) .+ NaN for _ = 1:2]; #store whether rollouts were successful

for t = 1:tmax #for each iteration
    agent_input = agent_input #copy to local variable
    world_state = world_state
    rew = rew
    h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step
    active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes
    agent_states[:, :, t] = world_state.agent_state' #store agent locations
    actions[:, t] = a[:] #store actions
    actions[ .~active, t] .= 0 #if finished
    a[.~active] .= 1 #no rollouts if finished

    at_rew = (.~(active .& exploit .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #don't perform rollout at goal location

    ahot = zeros(Float32, 5, batch); for b = 1:batch ahot[Int(a[b]), b] = 1f0 end #one-hot action representation
    #run an iteration of rollouts
    _, _, (path_cv, _, _, plan_states_cv) = planner.planning_algorithm(world_state,ahot,wall_environment.dimensions,agent_output,at_rew,planner,m,h_rnn,m.model_properties,returnall = true)

    exploit[rew[:] .> 0.5] .= true #exploitation phase
    #now run a separate model update step where we use a _different_ set of rollouts to guide behavior
    rew, agent_input, world_state, predictions = wall_environment.step(
                agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
            )
    path = reshape(world_state.planning_state.plan_input[1:(4*Lplan), :], 4, Lplan, batch) #set of actions
    plan_states = world_state.planning_state.plan_cache #set of states

    #store the results of our rollouts
    for b = 1:batch #for each episode
        if (world_state.planning_state.plan_cache[1, b] > 0.5) && (~at_rew[b]) #if we performed a rollout
            plans[b, t, :] = plan_states[:, b] #store state sequence
            plan_as[b, t, :] = [argmax(path[:, l, b]) for l = 1:Lplan] #store action sequence
	        for loc = 1:16 #for each location
                success[b, t, loc] = any(plan_states[:, b] .== loc) #store whether the rollout was successful w.r.t. this location
                success_cv[b, t, loc] = any(plan_states_cv[:, b] .== loc) #crossvalidated
            end
        end
    end

    rew[.~active] .= 0f0 #zero reward for finished episodes
    push!(all_rews, rew) #store rewards

end

wall_loc, ps = world_state.environment_state.wall_loc, world_state.environment_state.reward_location #wall and reward locations
rews = reduce(vcat, all_rews)' #combine into single array

##collect some summary data
println("collecting data")
next_as = zeros(batch, tmax); #next action at each timepoint
opt_as = zeros(batch, tmax, 4); #optimal action at each timepoint
agent_opt_as = zeros(batch, tmax) #did the agent take an optimal action
goal_steps = zeros(batch, tmax); #steps to next goal
goal_dist = zeros(batch, tmax); #optimal steps to next goal

trial_ts = zeros(batch, tmax); #iteration within trial
trial_ids = zeros(batch, tmax); #trial number
trial_anums = zeros(batch, tmax); #action number within trial
for b = 1:batch #for each episode
    Nrew = sum(rews[b, :] .> 0.5) #number of rewards
    sortrew = sortperm(-rews[b, :])
    rewts = sortrew[1:Nrew] #reward times
    diffs = [rewts; tmax+1] - [0; rewts] #durations of each trial
    trial_ids[b, :] = reduce(vcat, [ones(diffs[i]) * i for i = 1:(Nrew+1)])[1:tmax] #trial numbers
    trial_ts[b, :] = reduce(vcat, [1:diffs[i] for i = 1:(Nrew+1)])[1:tmax] #time within trial

    #zero out timepoints where episode is finished
    finished = findall(actions[b, :] .== 0)
    trial_ids[b, finished] .= 0
    trial_ts[b, finished] .= 0

    #collect physical action number within each trial
    ep_as = actions[b, :] #actions taken in this episode
    for id = 1:(Nrew+1) #for each trial
        inds = findall(trial_ids[b, :] .== id) #timepoints corresponding to this trial
        trial_as = ep_as[inds] #actions within this trial
        anums = zeros(Int64, length(inds)) #container for action numbers
        anum = 1 #start at 1
        for a = 2:length(inds) #iterate through the timesteps (although could've done the whole thing with a cumsum)
            anums[a] = anum #store action number
            if trial_as[a] <= 4.5 anum +=1 end #only increment if physical action
        end
        trial_anums[b, inds] = anums #store action numbers
    end
    trial_anums[b, finished] .= 0 #zero after episode is finished

    #get list of next actions
    for t = 1:(tmax) #for each iteration
        next = (actions[b, t:end][actions[b, t:end] .< 4.5]) #future physical actions
        if length(next) == 0 #last action
            next_as[b, t] = 0 #no next action
        else
            next_as[b, t] = next[1] #first future action
        end
    end
    #get distance to goal and optimal actions
    dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena) #all distances to goal
    for t = 1:tmax
        state = Int.(agent_states[b, :, t]) #current agent state
        goal_dist[b, t] = dists[state[1], state[2]] #minimum distance to goal
        opt_pi = optimal_policy(state, wall_loc[:, :, b:b], dists, ed) #optimal policy
        opt_as[b, t, :] = Float64.(opt_pi[1:4] .> 1e-2) #set of optimal actions
        nexta = Int(next_as[b,t]) #next action here
        if nexta > 0.5 agent_opt_as[b, t] = Float64.(opt_as[b, t, nexta] > 1e-2) end #was it optimal?
        goal_steps[b, t] = trial_anums[b, findall(trial_ids[b, :] .== trial_ids[b, t])[end]] - trial_anums[b, t]+1 #number of physical actions taken to goal
    end
end

# calculate wall conformity
println("calculating wall conformity")
function get_action(Larena, s1, s2) #find the action that would take us from state 1 to state 2
    ss = state_from_loc(Larena, [s1; s2]') #convert to index
    s1, s2 = ss[:, 1], ss[:, 2]
    vecs = [[1;0], [-1; 0], [0;1], [0; -1]] #possible actions
    for a = 1:4
        if all(s2 .== ((s1 + vecs[a] .+ Larena .- 1) .% Larena .+ 1)) #find the one that takes us to s2
            return a
        end
    end
    return 0
end

batch_wall_probs = [] #true probability of going through wall
batch_rand_wall_probs = [] #control probability
for b = 1:batch
    #println(b)
    plan_times = findall(plans[b, :, 1] .> 0.5) #indices at which the agent performed a rollout
    wall_probs = [] #for this episode
    rand_wall_probs = [] #for this episode
    for t = plan_times #go through rollouts
        #println(t)
        newplan = [state_ind_from_state(Larena, agent_states[b, :, t])[1]; plans[b, t, :]] #concatenate current state and subsequent imagined states
        newplan_as = plan_as[b, t, :] #what are the actions that were taken
        nsteps = sum(newplan .> 0.5)-1 #number of steps that were taken
        for i = 1:nsteps #for each step
            s1, s2 = newplan[i], newplan[i+1] #first and second state
            if s1 != s2 #check that we moved
                walls = world_state.environment_state.wall_loc[Int(newplan[i]), :, b] #wall locations in this episode
                a = get_action(Larena, s1, s2) #action that would've taken me there
                if a > 0.5 #ignore discontinuous jumps
                    wall_probs = [wall_probs; walls[Int(newplan_as[i])]] #did I move through a wall?
                    rand_wall_probs = [rand_wall_probs; mean(walls)] #baseline probability
                end
            end
        end
    end
    batch_wall_probs = [batch_wall_probs; wall_probs] #append result for this epsode
    batch_rand_wall_probs = [batch_rand_wall_probs; rand_wall_probs] #append baseline
end

#store result
res_dict[seed]["batch_wall_probs"] = batch_wall_probs
res_dict[seed]["batch_rand_wall_probs"] = batch_rand_wall_probs


# now look at fraction of successful replays
println("calculating success frequency")
true_succs, false_succs = [], [] #true and control success fractions
for b = 1:batch #for each episode
    inds = findall( .~ isnan.(success[b, :, rewlocs[b]]) ) #planning times
    true_succs = [true_succs; success[b, inds, rewlocs[b]]] #success of plans w.r.t true reward locations
    new_false = [mean(success[b, ind, findall(1:16 .!= rewlocs[b])]) for ind = inds] #w.r.t control locations
    false_succs = [false_succs; new_false] #store data
end
# save data
res_dict[seed]["true_succs"] = true_succs
res_dict[seed]["false_succs"] = false_succs


## now look at p(goal | plan number)
println("calculating success by replay number")
maxL = 5 #maximum number of plans to consider
for minL = [2;3] #consider a minimum number of rollouts
    succ_byp = zeros(batch, tmax, maxL + 2) .+ NaN; #container for storing results
    succ_byp_ctrl = zeros(15, batch, tmax, maxL + 2) .+ NaN; #container for storing control results
    minreps, minreps_ctrl = [], []
    for b = 1:batch #for each episode
        nt, np = 0, 0 #number of sequences and individual rollouts within the sequence
        loc = rewlocs[b] #reward location
        for t = 1:sum(actions[b, :] .> 0.5) #for each index before episode finished
            if isnan(success_cv[b, t, loc]) #no rollout
    	        np = 0 #reset
            else
    	        if np == 0 #first plan at this step
    	            nt += 1 #increment plan number
    	        end
    	        np += 1 #increment rollout number within sequence
                if (np <= (maxL+2))
                    succ_byp[b, nt, np] = success_cv[b, t, loc] #was this rollout successful
                    for i = 1:15 #ctrl locations
                        succ_byp_ctrl[i, b, nt, np] = success_cv[b, t, findall((1:16) .!= loc)[i]] #was is successgful to the control loc
                    end
                end
    	    end
        end
        newreps = succ_byp[b, 1:nt, :] #extract success data
        nplans = sum( .~ isnan.(newreps), dims = 2)[:] #number of rollouts within each sequence
        inds = findall( (nplans .>= minL) .& (nplans .<= maxL) ) #subselect by number of rollouts in sequence
        push!(minreps, newreps[inds, 1:minL]) #store success rates
        push!(minreps_ctrl, succ_byp_ctrl[:, b, inds, 1:minL]) #store controls
    end
    
    cat_succ = reduce(vcat, minreps); #combine arrays across episodes
    cat_succ_ctrl = reduce(hcat, minreps_ctrl); #combine array across episodes
    res_dict[seed]["suc_by_rep_min$minL"] = cat_succ; #store result
    res_dict[seed]["suc_by_rep_min$(minL)_ctrl"] = cat_succ_ctrl; #store ctrl result
end

## now look at p(follow | goal) and p(follow | no-goal)
println("calculating behavior by success/non-success")
succs, nons = [], []
for b = 1:batch #for each episode
    succ_inds = findall(success[b, :, rewlocs[b]] .== 1) #indices of successful rollouts
    non_inds = findall(success[b, :, rewlocs[b]] .== 0) #indices of unsuccessful rollouts
    succs = [succs; plan_as[b, succ_inds, 1] .== next_as[b, succ_inds]] #am I consistent with rollout actions after successful rollouts?
    nons = [nons; plan_as[b, non_inds, 1] .== next_as[b, non_inds]] #am I consistent after unssuccessful rollouts?
end

#write the data
res_dict[seed]["follow_succs"] = succs
res_dict[seed]["follow_non"] = nons

end

#now store all the data
@save datadir * "model_replay_analyses.bson" res_dict

