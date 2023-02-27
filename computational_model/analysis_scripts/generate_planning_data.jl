## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics
using BSON: @save

loss_hp = LossHyperparameters(0, 0, 0, 0, 0, 0, 1000, true, 0f0, () -> ())
Save = true

greedy_actions = true
no_planning = false
seeds = 61:65
epoch = plan_epoch

for seed = seeds
    fname = "N100_T50_seed$(seed)_Lplan8_$epoch"
    println("loading ", fname)
    network, opt, store, hps, policy, prediction = recover_model("../models/maze/$fname", modular = true);

    Larena = hps["Larena"]
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
    )
    m = ModularModel(model_properties, network, policy, prediction, forward_modular)
    Nstates = Larena^2

    ##
    Random.seed!(1)
    batch_size = 50000
    tic = time()
    L, ys, rews, as, world_states, hs = run_episode(
        m, wall_environment, loss_hp; batch=batch_size, calc_loss = false
    )
    states = reduce((a, b) -> cat(a, b, dims = 3), [ws.agent_state for ws = world_states])
    wall_loc, ps = world_states[1].environment_state.wall_loc, world_states[1].environment_state.reward_location
    Tmax = size(as, 2)
    rew_locs = reshape(ps, Nstates, batch_size, 1) .* ones(1, 1, Tmax) #for each time point
    println(sum(rews .> 0.5) / batch_size, " ", time() - tic)
    println("planning fraction: ", sum(as .> 4.5) / sum(as .> 0.5))

    ## collect some data
    plan_states = zeros(batch_size, Tmax, 8);
    plan_steps = zeros(batch_size, Tmax); #how many steps/actions were planned
    for t = 1:Tmax-1
        plan_states[:, t, :] = world_states[t+1].planning_state.plan_cache';
        plan_steps[:,t] = sum(plan_states[:, t, :] .> 0.5, dims = 2)[:];
    end

    trial_ts = zeros(batch_size, Tmax)
    trial_ids = zeros(batch_size, Tmax)
    trial_anums = zeros(batch_size, Tmax) #action number!
    for b = 1:batch_size
        Nrew = sum(rews[b, :] .> 0.5)
        sortrew = sortperm(-rews[b, :])
        rewts = sortrew[1:Nrew]
        diffs = [rewts; Tmax+1] - [0; rewts]
        trial_ids[b, :] = reduce(vcat, [ones(diffs[i]) * i for i = 1:(Nrew+1)])[1:Tmax]
        trial_ts[b, :] = reduce(vcat, [1:diffs[i] for i = 1:(Nrew+1)])[1:Tmax]

        finished = findall(as[b, :] .== 0)
        trial_ids[b, finished] .= 0
        trial_ts[b, finished] .= 0
        plan_steps[b, finished] .= 0

        ep_as = as[b, :]
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
    end

    ## look at performance by trial

    Rmin = 4
    inds = findall(sum(rews, dims = 2)[:] .>= Rmin) #finish at least Rmin trials
    perfs = reduce(hcat, [[trial_anums[b, trial_ids[b, :] .== t][end] for t = 1:Rmin] for b = inds])'
    ### compute optimal performance ###
    mean_dists = zeros(batch_size)
    shortest_dists = zeros(batch_size, Larena, Larena)
    for b in 1:batch_size
        dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena)
        mean_dists[b] = sum(dists) / (Nstates - 1)
        shortest_dists[b, :, :] = dists
    end
    println(mean(mean_dists), " ", std(mean_dists))
    μ, s = mean(perfs, dims = 1)[:], std(perfs, dims = 1)[:]/sqrt(batch_size)
    data = [Rmin, μ, s, mean(mean_dists)]
    Save && @save "$(datadir)/model_by_trial$(prior)$seed.bson" data

    ## planning by difficulty

    trials = 15
    new_RTs = zeros(trials, batch_size, hps["T"]) .+ NaN
    new_alt_RTs = zeros(trials, batch_size, hps["T"]) .+ NaN
    new_dists = zeros(trials, batch_size) .+ NaN
    for b = 1:batch_size
        #println(b)
        rew = rews[b, :] #rewards in this episode
        min_dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena) #minimum distances to goal for each state
        for trial = 2:trials
            #println(trial)
            if sum(rew .> 0.5) .> (trial - 0.5) #finish trial
                inds = findall((trial_ids[b, :] .== trial) .& (trial_ts[b, :] .> 1.5)) #all timepoints within trial

                anums = trial_anums[b, inds]
                RTs = [sum(anums .== anum) for anum = 1:anums[end]]

                plan_nums = plan_steps[b, inds]
                alt_RTs = [sum(plan_nums[anums .== anum]) for anum = 1:anums[end]] #count as number of simulated steps
                new_alt_RTs[trial, b, 1:length(alt_RTs)] = alt_RTs #reaction times

                for anum = 1:anums[end]
                    ainds = findall(anums .== anum)
                    if length(ainds) > 1.5
                        @assert all(plan_nums[ainds[1:(length(ainds)-1)]] .> 0.5) #should all have non-zero plans
                    end
                end

                new_RTs[trial, b, 1:length(RTs)] = RTs #reaction times
                state = states[:, b, inds[1]] #initial state
                new_dists[trial, b] = min_dists[Int(state[1]), Int(state[2])]
                if rand() < 1e-4
                    println(b, " ", new_dists[trial, b], " ", length(inds))
                end
            end
        end
    end

    dists = 1:8
    dats = [new_RTs[(new_dists.==dist), :] for dist in dists]
    data = [dists, dats]
    Save && @save "$(datadir)model_RT_by_complexity$(prior)$(seed)_$epoch.bson" data
    alt_dats = [new_alt_RTs[(new_dists.==dist), :] for dist in dists]
    data = [dists, alt_dats]
    Save && @save "$(datadir)model_RT_by_complexity_bystep$(prior)$(seed)_$epoch.bson" data

    ## look at exploration

    RTs = zeros(size(rews)) .+ NaN
    unique_states = zeros(size(rews)) .+ NaN #how many states had been seen when the action was taken
    for b = 1:batch_size
        inds = findall(trial_ids[b, :] .== 1)
        anums = Int.(trial_anums[b, inds])
        if sum(rews[b, :]) == 0 tmax = sum(as[b, :] .> 0.5) else tmax = findall(rews[b, :] .== 1)[1] end
        visited = Bool.(zeros(16)) #which states have been visited
        for anum = unique(anums)
            state = states[:,b,findall(anums .== anum)[1]]
            visited[Int(state_ind_from_state(Larena, state)[1])] = true
            unique_states[b, anum+1] = sum(visited)
            RTs[b, anum+1] = sum(anums .== anum)
        end
    end

    data = [RTs, unique_states]
    Save && @save "$(datadir)model_unique_states$(prior)_$(seed)_$epoch.bson" data

    ## do decoding of rew loc by unique states
    unums = 1:15
    dec_perfs = zeros(length(unums))
    for unum = unums
        inds = findall(unique_states .== unum)
        ahot = zeros(Float32, 5, length(inds))
        for (i, ind) = enumerate(inds) ahot[Int(as[ind]), i] = 1f0 end
        X = [hs[:, inds]; ahot] #Nhidden x batch x T -> Nhidden x iters
        Y = rew_locs[:, inds]
        Yhat = m.prediction(X)[18:33, :]
        Yhat = exp.(Yhat .- Flux.logsumexp(Yhat; dims=1)) #softmax over states
        perf = sum(Yhat .* Y) / size(Y, 2)
        #perf = mean(Y[argmax(Yhat[:, i]), i] for i = 1:size(Y, 2))

        println("reward ", unum, ": ", perf, "\n")
        dec_perfs[unum] = perf
    end
    data = [unums, dec_perfs]
    Save && @save "$(datadir)model_exploration_predictions$(prior)_$(seed)_$epoch.bson" data

    ## perturbation analysis
    run_perturbation = false
    if run_perturbation
        data = run_perturbation_analysis(m, hs, rew_locs, trial_ids, trial_ts, wall_environment, Nstates, hps, res = 1, niter = 1000)
        Save && @save "$(datadir)/perturbation_data_planning$(prior)$(seed)_$epoch.bson" data
    end
end
