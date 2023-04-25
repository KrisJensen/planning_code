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
world_state, agent_input = wall_environment.initialize(
    zeros(2), zeros(2), batch, m.model_properties
)
agent_state = world_state.agent_state
h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
rew = zeros(batch) #container for storing reward info

# initialize containers for storing data
tmax = 200 #number of iterations to run
exploit = Bool.(zeros(batch))
Lplan = model_properties.Lplan
plans, plans_cv = [zeros(batch, tmax, Lplan) .+ NaN for _ = 1:2];
plan_as, plan_as_cv = [zeros(batch, tmax, Lplan) .+ NaN for _ = 1:2];
planner, initial_plan_state = build_planner(Lplan, Larena)
agent_states = zeros(batch, 2, tmax);
actions = zeros(batch, tmax)
all_rews = []

rewlocs = [argmax(world_state.environment_state.reward_location[:, i]) for i = 1:batch]
success, success_cv = [zeros(batch, tmax, 16) .+ NaN for _ = 1:2];

for t = 1:tmax
    agent_input = agent_input
    world_state = world_state
    rew = rew
    h_rnn, agent_output, a = m.forward(m, ed, agent_input; h_rnn=h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)
    active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active heads
    agent_states[:, :, t] = world_state.agent_state'
    actions[:, t] = a[:]
    actions[ .~active, t] .= 0 #finished
    a[.~active] .= 1 #no planning
    ### explicitly run planning and return states!!! ###
    at_rew = (.~(active .& exploit .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #whether to store plan

    ###run crossvalidated planning and return states!!!
    ahot = zeros(Float32, 5, batch); for b = 1:batch ahot[Int(a[b]), b] = 1f0 end
    _, _, (path_cv, _, _, plan_states_cv) = planner.planning_algorithm(world_state,ahot,wall_environment.dimensions,agent_output,at_rew,planner,m,h_rnn,m.model_properties,returnall = true)

    exploit[rew[:] .> 0.5] .= true #exploitation phase
    rew, agent_input, world_state, predictions = wall_environment.step(
                agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
            )
    path = reshape(world_state.planning_state.plan_input[1:(4*Lplan), :], 4, Lplan, batch)
    plan_states = world_state.planning_state.plan_cache

    println(t, " ", mean(plan_states), " ", mean(path), " ", mean(at_rew))
    #store all plans!!!
    for b = 1:batch
        if (plan_states[1, b] > 0.5) && (~at_rew[b])
            plans[b, t, :] = plan_states[:, b]
            plan_as[b, t, :] = [argmax(path[:, l, b]) for l = 1:Lplan]
            plans_cv[b, t, :] = plan_states_cv[:, b]
            plan_as_cv[b, t, :] = [argmax(path_cv[:, l, b]) for l = 1:Lplan]
	        for loc = 1:16
                success[b, t, loc] = any(plan_states[:, b] .== loc)
                success_cv[b, t, loc] = any(plan_states_cv[:, b] .== loc)
            end
        end
    end

    rew[.~active] .= 0f0
    πt = exp.(agent_output[1:5, :])
    println(mean(maximum(πt, dims = 1)))
    push!(all_rews, rew)

end

wall_loc, ps = world_state.environment_state.wall_loc, world_state.environment_state.reward_location
println(mean(sum(all_rews)))
rews = reduce(vcat, all_rews)'

##collect some summary data
next_as = zeros(batch, tmax); #next action at each timepoint
opt_as = zeros(batch, tmax, 4); #optimal action at each timepoint
agent_opt_as = zeros(batch, tmax) #did the agent take an optimal action
goal_steps = zeros(batch, tmax); #steps to next goal
goal_dist = zeros(batch, tmax); #optimal steps to next goal

trial_ts = zeros(batch, tmax);
trial_ids = zeros(batch, tmax);
trial_anums = zeros(batch, tmax); #action number!
for b = 1:batch
    Nrew = sum(rews[b, :] .> 0.5)
    sortrew = sortperm(-rews[b, :])
    rewts = sortrew[1:Nrew]
    diffs = [rewts; tmax+1] - [0; rewts]
    trial_ids[b, :] = reduce(vcat, [ones(diffs[i]) * i for i = 1:(Nrew+1)])[1:tmax]
    trial_ts[b, :] = reduce(vcat, [1:diffs[i] for i = 1:(Nrew+1)])[1:tmax]

    finished = findall(actions[b, :] .== 0)
    trial_ids[b, finished] .= 0
    trial_ts[b, finished] .= 0

    ep_as = actions[b, :]
    for id = 1:(Nrew+1)
        inds = findall(trial_ids[b, :] .== id)
        trial_as = ep_as[inds]
        anums = zeros(Int64, length(inds))
        anum = 1
        for a = 2:length(inds)
            anums[a] = anum
            if trial_as[a] <= 4.5 anum +=1 end
        end
        trial_anums[b, inds] = anums
    end
    trial_anums[b, finished] .= 0

    #get list of next actions
    for t = 1:(tmax)
        next = (actions[b, t:end][actions[b, t:end] .< 4.5])
        if length(next) == 0
            next_as[b, t] = 0 #not defined
        else
            next_as[b, t] = next[1]
        end
    end
    #get distance to goal and optimal actions
    dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena)
    for t = 1:tmax
        state = Int.(agent_states[b, :, t])
	goal_dist[b, t] = dists[state[1], state[2]] #minimum distance to goal
        opt_pi = optimal_policy(state, wall_loc[:, :, b:b], dists, ed) #optimal policy
	opt_as[b, t, :] = Float64.(opt_pi[1:4] .> 1e-2) #optimal actions
	nexta = Int(next_as[b,t])
	if nexta > 0.5 agent_opt_as[b, t] = Float64.(opt_as[b, t, nexta] > 1e-2) end
	##remaining steps to goal ##
	goal_steps[b, t] = trial_anums[b, findall(trial_ids[b, :] .== trial_ids[b, t])[end]] - trial_anums[b, t]+1
    end
end

## plot wall conformity

function get_action(Larena, s1, s2)
    ss = state_from_loc(arena, [s1; s2]')
    s1, s2 = ss[:, 1], ss[:, 2]
    vecs = [[1;0], [-1; 0], [0;1], [0; -1]]
    for a = 1:4
        if all(s2 .== ((s1 + vecs[a] .+ Larena .- 1) .% Larena .+ 1))
            return a
        end
    end
    return 0
end

batch_wall_probs = []
batch_rand_wall_probs = []
for b = 1:batch
    plan_times = findall(plans[b, :, 1] .> 0.5)
    wall_probs = []
    rand_wall_probs = []
    for t = plan_times
        newplan = [state_ind_from_state(Larena, agent_states[b, :, t])[1]; plans[b, t, :]]
        newplan_as = plan_as[b, t, :]
        nsteps = sum(newplan .> 0.5)-1
        for i = 1:nsteps
            s1, s2 = newplan[i], newplan[i+1]
            if s1 != s2 #check that we moved
                walls = world_state.environment_state.wall_loc[Int(newplan[i]), :, b]
                a = get_action(Larena, s1, s2)
                if a > 0.5 #ignore discontinuous
                    wall_probs = [wall_probs; walls[Int(newplan_as[i])]] #did I move through a wall?
                    rand_wall_probs = [rand_wall_probs; mean(walls)] #baseline probability
                end
            end
        end
    end
    if length(wall_probs) >= 5
        batch_wall_probs = [batch_wall_probs; mean(wall_probs)]
        batch_rand_wall_probs = [batch_rand_wall_probs; mean(rand_wall_probs)]
    end
end

res_dict[seed]["batch_wall_probs"] = batch_wall_probs
res_dict[seed]["batch_rand_wall_probs"] = batch_rand_wall_probs

μ, s = mean(batch_wall_probs), std(batch_wall_probs)/sqrt(batch)
μr, sr = mean(batch_rand_wall_probs), std(batch_rand_wall_probs)/sqrt(batch)

## now look at fraction of successful replays
true_succs, false_succs = [], []
for b = 1:batch
    if sum( .~ isnan.(success[b, :, rewlocs[b]]) ) >= 5
        push!(true_succs, nanmean(success[b, :, rewlocs[b]]))
        false_inds = ones(16); false_inds[rewlocs[b]] = 0
        push!(false_succs, nanmean(success[b, :, Bool.(false_inds)]))
    end
end

res_dict[seed]["true_succs"] = true_succs
res_dict[seed]["false_succs"] = false_succs

μ, s = mean(true_succs), std(true_succs)/sqrt(length(true_succs))
μr, sr = mean(false_succs), std(false_succs)/sqrt(length(false_succs))

## now look at p(goal | plan number)
maxL = 5
for cv = [false; true]
if cv succ, cvstr = success_cv, "_cv" else succ, cvstr = success, "" end
for minL = [2;3]
    maxt = 50
    succ_byp = zeros(batch, maxt, maxL + 2) .+ NaN;
    succ_byp_ctrl = zeros(15, batch, maxt, maxL + 2) .+ NaN;
    minreps, minreps_ctrl = [], []
    for b = 1:batch
        nt, np = 0, 0
        loc = rewlocs[b]
        #loc = rand(1:16)
        for t = 1:tmax
            if isnan(succ[b, t, loc]) #no plan
    	        np = 0 #reset
            else
    	        if np == 0 #first plan at this step
    	            nt += 1 #increment plan number
    	        end
    	        np += 1 #increment plan step
                if (np <= (maxL+2)) && (nt <= maxt)
                    succ_byp[b, nt, np] = succ[b, t, loc]
                    for i = 1:15 #ctrl locations
                        succ_byp_ctrl[i, b, nt, np] = succ[b, t, findall((1:16) .!= loc)[i]]
                    end
                end
    	    end
        end
        newreps = succ_byp[b, 1:min(nt, maxt), :]
        nplans = sum( .~ isnan.(newreps), dims = 2)[:]
        inds = findall( (nplans .>= minL) .& (nplans .<= maxL) )
        push!(minreps, newreps[inds, 1:minL])
        push!(minreps_ctrl, succ_byp_ctrl[:, b, inds, 1:minL])
    end
    
    cat_succ = reduce(vcat, minreps);
    cat_succ_ctrl = reduce(hcat, minreps_ctrl);
    res_dict[seed]["suc_by_rep_min$minL$cvstr"] = cat_succ
    res_dict[seed]["suc_by_rep_min$(minL)_ctrl$cvstr"] = cat_succ_ctrl

    print(size(cat_succ))
    μ, s = mean(cat_succ, dims = 1)[:], std(cat_succ, dims = 1)[:] / sqrt(size(cat_succ, 1))
    for i = 1:minL dat = cat_succ[:, i] - cat_succ[:, 1]; println(mean(dat), " ", std(dat)/sqrt(length(dat))) end
    
    μ2 = mean(cat_succ_ctrl, dims = (1,2))[:]
    s2 = std(cat_succ_ctrl, dims = (1,2))[:] / sqrt(size(cat_succ_ctrl, 1)*size(cat_succ_ctrl, 2))
    
end
end

## now look at p(follow | goal) and p(follow | no-goal)

succs, nons = [], []
for b = 1:batch
    succ_inds, non_inds = findall(success[b, :, rewlocs[b]] .== 1), findall(success[b, :, rewlocs[b]] .== 0)
    println(length(succ_inds), " ", length(non_inds))
    if length(succ_inds) >= 3 && length(non_inds) >= 3
        push!(succs, mean(plan_as[b, succ_inds, 1] .== next_as[b, succ_inds]))
        push!(nons, mean(plan_as[b, non_inds, 1] .== next_as[b, non_inds]))
    end
end
res_dict[seed]["follow_succs"] = succs
res_dict[seed]["follow_non"] = nons
μ, s = mean(succs), std(succs)/sqrt(length(succs))
μr, sr = mean(nons), std(nons)/sqrt(length(nons))

end

@save datadir * "result_dict.bson" res_dict

