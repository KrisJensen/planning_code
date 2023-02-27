include("anal_utils.jl")
using BSON: @load
using BSON: @save
using ToPlanOrNotToPlan
using Statistics, NaNStatistics, LinearAlgebra, Random, HypothesisTests, Flux, Zygote, SQLite, DataFrames
using StatsBase

greedy_actions = true
prefix = ""
epoch = plan_epoch
N = 100; Lplan = 8

function repeat_actions(;seeds, prefix, epoch, prior, greedy_actions, N, Lplan)

game_type = "play"
loss_hp = LossHyperparameters(0, 0, 0, 0, 0, 0, 1000, true, 0, () -> ())

## load human data

@load "$(datadir)/human_RT_and_rews_play.bson" data; data_play = data
@load "$(datadir)/human_RT_and_rews_follow.bson" data; data_follow = data
med_RTs = [[nanmean(RTs) for RTs = data["all_RTs"]] for data = [data_play, data_follow]]
mean_rews = [[sum(rews)/size(rews, 1) for rews = data["all_rews"]] for data = [data_play, data_follow]]
keep = findall(med_RTs[2] .< 690)
Nkeep = length(keep)

@load "$(datadir)/human_all_data_play.bson" data
all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time = [dat[keep] for dat = data]

#load lognormal params
@load "$datadir/guided_lognormal_params_delta.bson" params
lognormal_params = Dict(key => params[key][keep, :] for key = ["initial"; "later"])

keep = 1:length(keep) #already subselected so keep everything from here onwards

## select participant

all_as_p, all_ys_p, all_pplans_p, all_Nplans_p = [], [], [], []
all_dists_to_rew_p, all_new_states_p = [], []
all_RTs_p = []

for i = keep
    println("user: $i")
    states = all_states[i]
    ps = all_ps[i]
    as = all_as[i]
    wall_loc = all_wall_loc[i]
    rews = all_rews[i]
    RTs = copy(all_RTs[i])
    trial_nums = all_trial_nums[i]
    trial_ts = all_trial_time[i]
    #dists = all_dists[i]
    batch_size = size(as, 1)

    ### compute thinking times! ###
    initial, later = [lognormal_params[key][i, :] for key = ["initial"; "later"]]
    #posterior mean for initial action
    initial_post_mean(r) = calc_post_mean(r, muhat=initial[1], sighat=initial[2], deltahat=initial[3])
    #posterior mean for later actions
    later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3])
    RTs[trial_ts .== 1] = initial_post_mean.(RTs[trial_ts .== 1]) #use different parameters for first action
    RTs[trial_ts .!= 1] = later_post_mean.(RTs[trial_ts .!= 1]) #posterior mean

    push!(all_RTs_p, RTs)

    ### load model

    as_p, ys_p, pplans_p, Nplans_p = [], [], [], []
    dists_to_rew_p, new_states_p = [], []
    for seed = seeds
        #println("seed: $seed")
        fname = "$(prefix)N$(N)_T50_seed$(seed)_Lplan$(Lplan)$(prior)_$epoch"
        println(fname)
        network, opt, store, hps, policy, prediction = recover_model("../models/maze/$fname", modular = true)
        model_properties, wall_environment, model_eval = build_environment(
            hps["Larena"], hps["Nhidden"], hps["T"], Lplan = hps["Lplan"],
            greedy_actions = greedy_actions
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        environment = wall_environment
        ed = environment.dimensions; Nout = m.model_properties.Nout
        Nhidden = m.model_properties.Nhidden; Nstates = ed.Nstates; T = ed.T

        ## run through same states

        ### initialize reward probabilities and state ###

        for rep = 1:21
            #println(rep)
            Random.seed!(rep)
            world_state, agent_input = environment.initialize(
                ps, Int32.(states[:, :, 1]), batch_size, m.model_properties
            )
            agent_state = world_state.agent_state
            world_state.environment_state.wall_loc .= copy(wall_loc)

            h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch_size)) #expand hidden state

            new_env_states = []
            new_ys = []
            new_as = []
            ts = zeros(Int64, batch_size) #timestep
            rew = zeros(batch_size)
            rew_prev = Bool.(zeros(batch_size))
            p_plans = zeros(batch_size, 51) .+ NaN
            N_plans = zeros(batch_size, 51)
            dists_to_rew = zeros(batch_size, 51) #distance to goal
            new_states = ones(2, batch_size, 100)
            new_rews = zeros(batch_size, 51)
            a5s = Bool.(zeros(batch_size))
            active = Bool.(ones(batch_size))
            nplan = 0
            while any(world_state.environment_state.time .< (T+1 - 1e-2))
                #println(maximum(world_state.environment_state.time), " ", maximum(ts))
                push!(new_env_states, world_state)
                #agent_input is Nin x batch
                h_rnn, agent_output, a = m.forward(m, ed, agent_input; h_rnn=h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)

                for b = 1:batch_size
                    if (rew[b] .< 0.5)
                        if ((~a5s[b]) && active[b]) ts[b] += 1 end #number of actual actions
                        a[1, b] = max(1, as[b, ts[b]])
                        if (as[b, ts[b]] > 0.5) && (~a5s[b]) && active[b] #write data if we didn't just plan
                            p_plans[b, ts[b]] = exp(agent_output[5, b])
                            new_states[:, b, ts[b]] = world_state.agent_state[:, b]
                            dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], hps["Larena"])
                            dists_to_rew[b, ts[b]] = dists[Int(new_states[1, b, ts[b]]), Int(new_states[2, b, ts[b]])]
                            #if rew_prev[b] #plan if we just teleported
                            if (ts[b] > 1.5) && (RTs[b, ts[b]] > -10000) #plan every action
                                if rand() < exp(agent_output[5, b]) nplan = 1 else nplan = 0 end #on policy
                            end
                        else
                            nplan -= 1
                            if rand() < exp(agent_output[5, b]) nplan = 1 else nplan = 0 end #on policy
                        end
                    else
                        #a[1, b] = Int32(1)
                        nplan = 0
                    end #use human action where we're not at reward

                    if nplan > 0.5 #if we're still planning
                        a[1, b] = Int32(5)
                        a5s[b] = true #still planning
                        if active[b] N_plans[b, ts[b]] += 1 end
                    else
                        a5s[b] = false
                    end
                end

                active = [(world_state.environment_state.time[b] < (T+1 - 1e-2)) & (as[b, ts[b]] .> 0.5) for b = 1:batch_size] #active heads
                rew_prev = (rew[:] .> 0.5)
                rew, agent_input, world_state, predictions = environment.step(
                        agent_output, a, world_state, environment.dimensions, m.model_properties,
                        m, h_rnn
                    )

                if ts[1] % 1000 == 0
                    println(ts)
                    println(ts[10], " ", a[10], " ", rew[10], " ", active[10], " ", world_state.environment_state.time[10])
                end

                for b = findall(rew_prev) #for the ones where we found reward before, teleport to correct location
                    #println(size(states), " ", b, " ", ts[b])
                    #println(states[:, b, ts[b]+1])
                    #if b == 10 println(ts[b]); println(states[:, b, ts[b]+1]) end
                    world_state.agent_state[:, b] .= states[:, b, ts[b]+1]
                end

                ### generate input
                ahot = zeros(Float32, 5, batch_size); for b = 1:batch_size ahot[Int(a[b]), b] = 1f0 end
                agent_input = Float32.(gen_input(world_state, ahot, rew, environment.dimensions, m.model_properties))

                rew, agent_input, a = zeropad_data(rew, agent_input, a, active)
                push!(new_ys, agent_output)
                push!(new_as, a)

            end

            push!(ys_p, reduce((a, b) -> cat(a, b, dims = 3), new_ys))
            push!(as_p, reduce(vcat, new_as)'); push!(dists_to_rew_p, dists_to_rew)
            push!(pplans_p, p_plans); push!(new_states_p, new_states); push!(Nplans_p, N_plans)
        end
    end
    push!(all_ys_p, ys_p); push!(all_as_p, as_p); push!(all_pplans_p, pplans_p); push!(all_Nplans_p, Nplans_p)
    push!(all_new_states_p, new_states_p); push!(all_dists_to_rew_p, dists_to_rew_p)
end

## quantify

for trial_type = ["explore"; "exploit"]
if trial_type == "explore" trialstr = "_explore" else trialstr = "" end
valid_users = 1:length(all_pplans_p)
alldat1, alldat2, alldat3, alldat4 = [], [], [], []
allres, allsims, allsims_s = [], zeros(length(valid_users), 3), zeros(length(valid_users), 3)
p_plans_by_u, RTs_by_u, dists_by_u, steps_by_u, N_plans_by_u, anums_by_u, trialnums_by_u = [], [], [], [], [], [], []

for i = 1:length(valid_users)
    #println(i)
    p_plans = mean(reduce((a, b) -> cat(a, b, dims = 3), all_pplans_p[i]), dims = 3)[:, :, 1]
    N_plans = mean(reduce((a, b) -> cat(a, b, dims = 3), all_Nplans_p[i]), dims = 3)[:, :, 1]

    cors = []
    cors_d = []
    cors_t = []

    #@assert all(all_new_states_p[i][1] .== all_new_states_p[i][end]) #across trials

    as, trial_ts, states, rews, trial_nums = all_as[i], all_trial_time[i], all_states[i], all_rews[i], all_trial_nums[i]
    dists_to_rew, new_states, RTs = all_dists_to_rew_p[i][1], all_new_states_p[i][1], all_RTs_p[i]

    sim = cor

    dat1, dat2, dat3, dat4, new_trial_nums, new_anums, Nplan_dat = [], [], [], [], [], [], []
    for b = 1:size(as, 1)
        #println(b)
        tmin = sortperm(-rews[b, :])[1]+1
        tmin = 2 #ignore very first action
        tmax = min(sum(as[b, :] .> 0.5), sum(p_plans[b, :] .> 0.0))
        if (tmax > tmin+5) && (sum(rews[b, :]) > 0.5)
            push!(cors, sim(log.(p_plans[b, tmin:tmax]), RTs[b, tmin:tmax]))
            push!(cors_d, sim(dists_to_rew[b, tmin:tmax], RTs[b, tmin:tmax]))
            push!(cors_t, sim(1 ./ trial_ts[b, tmin:tmax], RTs[b, tmin:tmax]))
            #push!(cors_d, cor(all_dists[i][b, tmin:tmax], RTs[b, tmin:tmax]))
            @assert all(new_states[:, b, tmin:tmax] .== states[:, b, tmin:tmax])
            @assert all(all_new_states_p[i][1][:, b, tmin:tmax] .== all_new_states_p[i][end][:, b, tmin:tmax]) #across trials
            append!(dat1, RTs[b, tmin:tmax]); append!(dat2, (p_plans[b, tmin:tmax]))
            append!(dat3, dists_to_rew[b, tmin:tmax]); append!(dat4, -trial_ts[b, tmin:tmax])
            append!(Nplan_dat, N_plans[b, tmin:tmax])
            new_trial_nums = [new_trial_nums; trial_nums[b,tmin:tmax]]
            new_anums = [new_anums; tmin:tmax]
        end
    end
    #dat1 = sqrt.(dat1)
    if trial_type == "exploit"
        inds = findall( (dat1 .< 30000) .&& (new_trial_nums .> 1.5) )
    else
        inds = findall( (dat1 .< 30000) .&& (new_trial_nums .< 1.5) )
    end
    dat1, dat2, dat3, dat4 = [dat[inds] for dat = [dat1, dat2, dat3, dat4]]
    append!(alldat1, dat1); append!(alldat2, dat2); append!(alldat3, dat3); append!(alldat4, dat4)
    push!(RTs_by_u, dat1); push!(p_plans_by_u, dat2); push!(dists_by_u, dat3); push!(steps_by_u, dat4);
    push!(N_plans_by_u, Nplan_dat[inds]); push!(anums_by_u, new_anums[inds]); push!(trialnums_by_u, new_trial_nums[inds])

    sim2(x1, x2) = cor(x1, x2)#calc_inf(Float64.(x1), Float64.(x2), noise = 1)
    sim3(x1, x2) = corspearman(Float64.(x1), Float64.(x2))

    s2, s3, s4 = sim2(dat1, dat2), sim2(dat1, dat3), sim2(dat1, dat4)
    s2_s, s3_s, s4_s = sim3(dat1, dat2), sim3(dat1, dat3), sim3(dat1, dat4)

    println("\nmodel$i: ", mean(cors), " ", std(cors), " ", s2)
    println("dist$i: ", mean(cors_d), " ", std(cors_d), " ", s3)
    println("time$i: ", mean(cors_t), " ", std(cors_t), " ", s4)

    X = Float64.([dat3 dat4 ones(length(dat3))])
    W = ToPlanOrNotToPlan.ridge(X, dat1, 1e-5)
    residuals = dat1 - X*W
    println("residual$i: ", cor(dat2, residuals))
    append!(allres, cor(residuals, dat2))
    allsims[i, :] = [s2; s3; s4]
    allsims_s[i, :] = [s2_s; s3_s; s4_s]
end

println("\nmodel correlation: ", cor(alldat1, alldat2))
println("by user: ", mean(allsims, dims = 1)[:], " ", std(allsims, dims = 1)[:]/sqrt(length(valid_users)))
println("residual correlation: ", mean(allres), " ", std(allres)/sqrt(length(allres)))

bins = 0.1:0.05:min(0.7, maximum(alldat2))
xs = 0.5*(bins[1:length(bins)-1] + bins[2:end])
dat = [alldat1[(alldat2 .>= bins[i]) .& (alldat2 .< bins[i+1])] for i = 1:length(bins)-1]
m = [mean(d) for d = dat]
s = [std(d)/sqrt(length(d)) for d = dat]
n = [length(d) for d = dat]
figure()
errorbar(xs, m, yerr = s, fmt = "k-")
savefig("$figdir/human_behavior/model_RT_prediction$trialstr.png", bbox_inches = "tight")
close()

data = Dict("residuals" => allres, "correlations" => allsims, "RTs" => alldat1, "pplans" => alldat2,
            "dists" => alldat3, "steps" => alldat4, "spearman" => allsims_s,
            "RTs_by_u" => RTs_by_u, "pplans_by_u" => p_plans_by_u,
            "dists_by_u" => dists_by_u, "steps_by_u" => steps_by_u,
            "N_plans_by_u" => N_plans_by_u, "N_plans" => reduce(vcat, N_plans_by_u),
            "trial_nums_by_u" => trialnums_by_u, "anums_by_u" => anums_by_u)

savename = "$(prefix)N$(N)_Lplan$(Lplan)$(prior)$(trialstr)_$epoch"
@save "$datadir/RT_predictions_$savename.bson" data
end

end

run = true
run && repeat_actions(;seeds, prefix, epoch, prior, greedy_actions, N, Lplan)
