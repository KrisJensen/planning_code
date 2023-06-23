# in this script, we clamp the agent trajectory to each human trajectory.
# we then evaluate pi(rollout) in each situation for comparison with human data.

#load some libraries
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics

try
    println("running default analyses: ", run_default_analyses)
catch e
    global run_default_analyses = true
end

"""
    repeat_actions(;seeds, prefix, epoch, N, Lplan)
This function clamps agents to human trajectories and stores data on the resulting policy
"""
function repeat_human_actions(;seeds, N, Lplan, epoch, prefix="")
println("analysing rollout probabilities when repeating human actions")

## load human data
@load "$(datadir)/human_RT_and_rews_play.bson" data; data_play = data
@load "$(datadir)/human_RT_and_rews_follow.bson" data; data_follow = data
mean_RTs = [[nanmean(RTs) for RTs = data["all_RTs"]] for data = [data_play, data_follow]] #mean RT for each participant
keep = findall(mean_RTs[2] .< 690)
Nkeep = length(keep) #only keep participants with mean RT < 690 ms during follow

@load "$(datadir)/human_all_data_play.bson" data #load all data from these participants
all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time = [dat[keep] for dat = data];

#load response time prior parameters
@load "$datadir/guided_lognormal_params_delta.bson" params
lognormal_params = Dict(key => params[key][keep, :] for key = ["initial"; "later"])

#initialize some results arrays
all_as_p, all_ys_p, all_pplans_p, all_Nplans_p = [], [], [], []
all_dists_to_rew_p, all_new_states_p = [], []
all_RTs_p = []

for i = 1:length(keep) #for each participant
    println("user: $i")
    #load data for this participant
    states = all_states[i] #physical locations
    ps = all_ps[i] #reward locations
    as = all_as[i] #actions taken
    wall_loc = all_wall_loc[i] #wall locations
    rews = all_rews[i] #rewards
    RTs = copy(all_RTs[i]) #make sure to copy to avoid mutating data
    trial_ts = all_trial_time[i] #iteration within trial
    batch_size = size(as, 1) #number of episodes

    ### compute thinking times from prior ###
    initial, later = [lognormal_params[key][i, :] for key = ["initial"; "later"]]
    #posterior mean for initial action
    initial_post_mean(r) = calc_post_mean(r, muhat=initial[1], sighat=initial[2], deltahat=initial[3])
    #posterior mean for later actions
    later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3])
    RTs[trial_ts .== 1] = initial_post_mean.(RTs[trial_ts .== 1]) #use different parameters for first action
    RTs[trial_ts .!= 1] = later_post_mean.(RTs[trial_ts .!= 1]) #posterior mean

    push!(all_RTs_p, RTs) #store RTs

    # now run all RL agents through this participant
    as_p, ys_p, pplans_p, Nplans_p = [], [], [], [] #arrays for storing data
    dists_to_rew_p, new_states_p = [], []
    for seed = seeds #for each model
        fname = "N$(N)_T50_Lplan$(Lplan)_seed$(seed)_$epoch" #model to load
        #println("loading ", fname)
        network, opt, store, hps, policy, prediction = recover_model(loaddir*fname) #load model parameters

        #construct RL environment and instantiate agent
        model_properties, wall_environment, model_eval = build_environment(
            hps["Larena"], hps["Nhidden"], hps["T"], Lplan = hps["Lplan"],
            greedy_actions = greedy_actions
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        # extract some useful parameters
        environment = wall_environment
        ed = environment.dimensions; Nout = m.model_properties.Nout
        Nhidden = m.model_properties.Nhidden; Nstates = ed.Nstates; T = ed.T

        for rep = 1:21 #repeat multiple times since rollouts are stochastic
            Random.seed!(rep)

            #initialize environment
            world_state, agent_input = environment.initialize(
                ps, Int32.(states[:, :, 1]), batch_size, m.model_properties
            )
            agent_state = world_state.agent_state #agent location
            world_state.environment_state.wall_loc .= copy(wall_loc) #wall location

            h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch_size)) #expand hidden state

            new_ys = []
            new_as = []
            ts = zeros(Int64, batch_size) #current timestep
            rew = zeros(batch_size) #current reward
            rew_prev = Bool.(zeros(batch_size)) #reward on previous iteration
            p_plans = zeros(batch_size, 51) .+ NaN #probability of performing a rollout
            N_plans = zeros(batch_size, 51) #number of rollouts performed
            dists_to_rew = zeros(batch_size, 51) #distance to goal
            new_states = ones(2, batch_size, 100) #agent states on each iteration
            new_rews = zeros(batch_size, 51) #reward on each iteration
            a5s = Bool.(zeros(batch_size)) #did we perform a rollout
            active = Bool.(ones(batch_size)) #has the episode finished
            nplan = 0
            while any(world_state.environment_state.time .< (T+1 - 1e-2)) #continue until time runs out for human or agent

                #forward pass of the RL agent
                h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step

                for b = 1:batch_size #for each episode
                    if (rew[b] .< 0.5) #did not get reward

                        if ((~a5s[b]) && active[b]) ts[b] += 1 end #index of human action to take
                        a[1, b] = max(1, as[b, ts[b]]) #overwrite with human action

                        if (as[b, ts[b]] > 0.5) && (~a5s[b]) && active[b] #write data if we previously took an action
                            p_plans[b, ts[b]] = exp(agent_output[5, b]) #rollout probability
                            new_states[:, b, ts[b]] = world_state.agent_state[:, b] #where are we now?
                            dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], hps["Larena"]) #distance to goal
                            dists_to_rew[b, ts[b]] = dists[Int(new_states[1, b, ts[b]]), Int(new_states[2, b, ts[b]])] #from here
                        end

                        if (ts[b] > 1.5) && (rand() < exp(agent_output[5, b]))
                            nplan = 1 #perform a rollout with probability pi(rollout) [the code supports enforcing multiple rollouts]
                        else
                            nplan = 0 #else just take the action
                        end
                    else #did get reward
                        nplan = 0 #no planning
                    end

                    if nplan > 0.5 #if we're performing a rollouts
                        a[1, b] = Int32(5) #set our action to 'rollout'
                        a5s[b] = true #just performed a rollout
                        if active[b] N_plans[b, ts[b]] += 1 end #if episode has not terminated, count the rollout
                    else
                        a5s[b] = false #just took a physical action
                    end
                end

                active = [(world_state.environment_state.time[b] < (T+1 - 1e-2)) & (as[b, ts[b]] .> 0.5) for b = 1:batch_size] #active episodes
                rew_prev = (rew[:] .> 0.5) # did we just find a reward
                #update environment with action 'a'
                rew, agent_input, world_state, predictions = environment.step(
                        agent_output, a, world_state, environment.dimensions, m.model_properties,
                        m, h_rnn
                    )

                for b = findall(rew_prev) #for the episodes where we found reward before
                    world_state.agent_state[:, b] .= states[:, b, ts[b]+1] #teleport to correct location
                end

                #generate input
                ahot = zeros(Float32, 5, batch_size); for b = 1:batch_size ahot[Int(a[b]), b] = 1f0 end #1hot action array
                agent_input = Float32.(gen_input(world_state, ahot, rew, environment.dimensions, m.model_properties))

                rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #remove data from finished episodes
                push!(new_ys, agent_output) #store new outputs
                push!(new_as, a) #store new actions

            end

            #store various pieces of data from this repetition
            push!(ys_p, reduce((a, b) -> cat(a, b, dims = 3), new_ys))
            push!(as_p, reduce(vcat, new_as)'); push!(dists_to_rew_p, dists_to_rew)
            push!(pplans_p, p_plans); push!(new_states_p, new_states); push!(Nplans_p, N_plans)
        end
    end
    #store data from this participant
    push!(all_ys_p, ys_p); push!(all_as_p, as_p); push!(all_pplans_p, pplans_p); push!(all_Nplans_p, Nplans_p)
    push!(all_new_states_p, new_states_p); push!(all_dists_to_rew_p, dists_to_rew_p)
end

# perform some analyses with the data collected above
for trial_type = ["explore"; "exploit"] #consider exploration and exploitation trials separately
if trial_type == "explore" trialstr = "_explore" else trialstr = "" end

# initialize some data array
alldRTs, alldplans, allddist, alldts = [], [], [], []
allres, allsims, allsims_s = [], zeros(length(all_pplans_p), 3), zeros(length(all_pplans_p), 3)
p_plans_by_u, RTs_by_u, dists_by_u, steps_by_u, N_plans_by_u, anums_by_u, trialnums_by_u = [], [], [], [], [], [], []

for i = 1:length(all_pplans_p) #iterate through users

    p_plans = mean(reduce((a, b) -> cat(a, b, dims = 3), all_pplans_p[i]), dims = 3)[:, :, 1] #rollout probabilities
    N_plans = mean(reduce((a, b) -> cat(a, b, dims = 3), all_Nplans_p[i]), dims = 3)[:, :, 1] #number of rollouts

    #extract data from this user
    as, trial_ts, states, rews, trial_nums = all_as[i], all_trial_time[i], all_states[i], all_rews[i], all_trial_nums[i]
    dists_to_rew, new_states, RTs = all_dists_to_rew_p[i][1], all_new_states_p[i][1], all_RTs_p[i]

    dRTs, dplans, ddist, dts, new_trial_nums, new_anums, Nplan_dat = [], [], [], [], [], [], []
    for b = 1:size(as, 1)
        tmin = 2 #ignore very first action
        tmax = min(sum(as[b, :] .> 0.5), sum(p_plans[b, :] .> 0.0)) #last action in episode
        if (tmax > tmin+5) && (sum(rews[b, :]) > 0.5)
            @assert all(new_states[:, b, tmin:tmax] .== states[:, b, tmin:tmax]) #check that we followed correct states
            @assert all(all_new_states_p[i][1][:, b, tmin:tmax] .== all_new_states_p[i][end][:, b, tmin:tmax]) #across trials
            #store various pieces of data from this participant
            append!(dRTs, RTs[b, tmin:tmax]); append!(dplans, (p_plans[b, tmin:tmax]))
            append!(ddist, dists_to_rew[b, tmin:tmax]); append!(dts, -trial_ts[b, tmin:tmax])
            append!(Nplan_dat, N_plans[b, tmin:tmax])
            new_trial_nums = [new_trial_nums; trial_nums[b,tmin:tmax]] #trial numbers
            new_anums = [new_anums; tmin:tmax] #global action number
        end
    end
    if trial_type == "exploit"
        inds = findall( (new_trial_nums .> 1.5) ) # exploitation trials
    else
        inds = findall( (new_trial_nums .< 1.5) ) #exploration trials
    end
    dRTs, dplans, ddist, dts = [dat[inds] for dat = [dRTs, dplans, ddist, dts]] #subselect based on trials of interest
    #append data to global data structures
    push!(RTs_by_u, dRTs); push!(p_plans_by_u, dplans); push!(dists_by_u, ddist); push!(steps_by_u, dts);
    push!(N_plans_by_u, Nplan_dat[inds]); push!(anums_by_u, new_anums[inds]); push!(trialnums_by_u, new_trial_nums[inds])

    #compute some correlations
    s_pplan, s_dist, s_ts = cor(dRTs, dplans), cor(dRTs, ddist), cor(dRTs, dts)
    allsims[i, :] = [s_pplan; s_dist; s_ts]
end

println("by user: ", mean(allsims, dims = 1)[:], " ", std(allsims, dims = 1)[:]/sqrt(size(allsims, 1)))

# let's just save a lot of data and we can select what to plot later
data = Dict("correlations" => allsims, "RTs_by_u" => RTs_by_u, "pplans_by_u" => p_plans_by_u,
            "dists_by_u" => dists_by_u, "steps_by_u" => steps_by_u,
            "N_plans_by_u" => N_plans_by_u, "N_plans" => reduce(vcat, N_plans_by_u),
            "trial_nums_by_u" => trialnums_by_u, "anums_by_u" => anums_by_u)

# save data for later loading
savename = "$(prefix)N$(N)_Lplan$(Lplan)$(trialstr)_$epoch"
@save "$datadir/RT_predictions_$savename.bson" data
end

end

#run_default_analyses is a global parameter in anal_utils.jl
run_default_analyses && repeat_human_actions(;seeds, N, Lplan, epoch) #call analysis function
