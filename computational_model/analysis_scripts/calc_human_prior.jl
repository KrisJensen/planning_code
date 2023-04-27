## load scripts and model
include("anal_utils.jl")
using BSON: @load, @save
using ToPlanOrNotToPlan
using Random, NaNStatistics, Statistics

@load "$(datadir)/human_all_data_follow.bson" data
_, _, _, _, _, all_RTs_f, all_trial_nums_f, all_trial_time_f = data;
Nuser = length(all_RTs_f)
delta_str = "_delta"
params = Dict(key => zeros(Nuser, 3) for key = ["initial"; "later"]) #initial true/false, all users, 3 parameters
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
        println(initial, " ", u, " ", length(RTs_f))

        #try different deltas
        deltas = 0:1:(minimum(RTs_f)-1)
        Ls, mus, sigs = [zeros(length(deltas)) for _ = 1:3]
        for (i, delta) = enumerate(deltas) #compute likelihood with each delta
            mus[i] = mean(log.(RTs_f .- delta))
            sigs[i] = std(log.(RTs_f .- delta))
            Ls[i] = sum(log.(lognorm.(RTs_f, mu = mus[i], sig = sigs[i], delta = delta)))
        end

        #extract ML parameters
        muhat, sighat, deltahat = [arr[argmax(Ls)] for arr = [mus, sigs, deltas]]
        params[key][u, :] = [muhat; sighat; deltahat] #store parameters
    end
end

@save "$datadir/guided_lognormal_params$delta_str.bson" params #mu, sigma, delta

