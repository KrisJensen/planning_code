## load scripts and model
include("anal_utils.jl")
using BSON: @load, @save
using ToPlanOrNotToPlan
using Random, NaNStatistics, Statistics
using ImageFiltering, StatsBase, Distributions

@load "$(datadir)/human_all_data_follow.bson" data
_, _, _, _, _, all_RTs_f, all_trial_nums_f, all_trial_time_f = data;
Nuser = length(all_RTs_f)
delta_str = ""
for learn_delta = [true, false]
    params = Dict(key => zeros(Nuser, 3) for key = ["initial"; "later"]) #initial true/false, all users, 3 parameters
    if learn_delta delta_str = "_delta" else delta_str = "" end
    for (i_init, initial) = enumerate([true, false])
        if initial key = "initial" else key = "later" end
        for u = 1:Nuser

            # datapoints during guided trials
            if initial
                inds = (all_trial_nums_f[u] .> 1.5) .& (all_trial_time_f[u] .== 1)
            else
                inds = (all_trial_nums_f[u] .> 1.5) .& (all_trial_time_f[u] .> 1.5)
            end
            RTs_f = all_RTs_f[u][inds]
            RTs_f = RTs_f[.~isnan.(RTs_f)]
            println(learn_delta, " ", initial, " ", u, " ", length(RTs_f))

            if learn_delta #try different deltas
                deltas = 0:1:(minimum(RTs_f)-1)
            else
                deltas = 0:0
            end

            Ls, mus, sigs = [zeros(length(deltas)) for _ = 1:3]
            for (i, delta) = enumerate(deltas) #compute likelihood with each delta
                mus[i] = mean(log.(RTs_f .- delta))
                sigs[i] = std(log.(RTs_f .- delta))
                Ls[i] = sum(log.(lognorm.(RTs_f, mu = mus[i], sig = sigs[i], delta = delta)))
            end

            #extract ML parameters
            muhat, sighat, deltahat = [arr[argmax(Ls)] for arr = [mus, sigs, deltas]]
            params[key][u, :] = [muhat; sighat; deltahat] #store parameters

            ### plot prior fit ###
            p_delay(x) = lognorm(x, mu = muhat, sig = sighat, delta = deltahat)
            xmin, xmax = deltahat+1, maximum(RTs_f)*1.05
            xmin = 0
            xs = xmin:5:xmax
            ps = p_delay.(xs)
            if initial bins = xmin:50:xmax else bins = xmin:30:xmax end
            figure()
            hist(RTs_f, bins = bins, color = "k", density = true, label = "empirical")
            plot(xs, ps, "b-", label = "model fit")
            xlabel("delay")
            ylabel("probability")
            legend(frameon = false)
            if initial
                savefig("./figs/weiji/lognormal_prior_initial_u$(u)$delta_str.png", bbox_inches = "tight")
            else
                savefig("./figs/weiji/lognormal_prior_u$(u)$delta_str.png", bbox_inches = "tight")
            end
            close()
        end
    end

    @save "$datadir/guided_lognormal_params$delta_str.bson" params #mu, sigma, delta
end


### count how often we run into trouble ###

# @load "$datadir/guided_lognormal_params_delta.bson" params #mu, sigma, delta
# @load "$(datadir)/human_all_data_play.bson" data
# _, _, _, _, _, all_RTs_p, all_trial_nums_p, all_trial_time_p = data;
# tots, bads = zeros(2, 100), zeros(2, 100)
# for u = 1:100
#     for (i, key) = enumerate(["initial"; "later"])
#         if key == "initial"
#             inds = (all_trial_nums_p[u] .> 0.5) .& (all_trial_time_p[u] .== 1)
#         else
#             inds = (all_trial_nums_p[u] .> 0.5) .& (all_trial_time_p[u] .> 1.5)
#         end
#         RTs = all_RTs_p[u][inds]
#         tots[i,u] = length(RTs)
#         bads[i,u] = sum(RTs .< params[key][u,3]+1)
#     end
# end
# println(sum(bads)/sum(tots))
