## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan

wrapstr = ""
wrapstr = "_euclidean"

println("computing prior parameters for human response times")

#start by loading our processed human data for the guided ('follow') trials
@load "$(datadir)/human_all_data_follow$wrapstr.bson" data
_, _, _, _, _, all_RTs_f, all_trial_nums_f, all_trial_time_f = data;
Nuser = length(all_RTs_f) #number of users

params = Dict(key => zeros(Nuser, 3) for key = ["initial"; "later"]) #all participants; initial/later actions; 3 parameters
for (i_init, initial) = enumerate([true, false]) #are we considering the first action in each trial?
    if initial key = "initial" else key = "later" end
    for u = 1:Nuser #for each participant

        # datapoints during guided trials
        if initial
            inds = (all_trial_nums_f[u] .> 1.5) .& (all_trial_time_f[u] .== 1) #first action in exploitation trials
        else
            inds = (all_trial_nums_f[u] .> 1.5) .& (all_trial_time_f[u] .> 1.5) #later actions in exploitation trials
        end
        RTs_f = all_RTs_f[u][inds] #reaction times for these actions
        RTs_f = RTs_f[.~isnan.(RTs_f)] #remove if there is missing data
        if u % 10 == 0 println("user $u, $key actions, $(length(RTs_f)) datapoints") end

        #try different deltas in our shifted lognormal prior
        deltas = 0:1:(minimum(RTs_f)-1) #list of deltas to try (the ones with appropriate support)
        Ls, mus, sigs = [zeros(length(deltas)) for _ = 1:3] #corresponding log liks and optimal params
        for (i, delta) = enumerate(deltas) #compute likelihood with each delta
            mus[i] = mean(log.(RTs_f .- delta)) #mean of the shifted lognormal
            sigs[i] = std(log.(RTs_f .- delta)) #standard deviation
            Ls[i] = sum(log.(lognorm.(RTs_f, mu = mus[i], sig = sigs[i], delta = delta))) #log likelihood of the data
        end

        #extract maximum likelihood parameters
        muhat, sighat, deltahat = [arr[argmax(Ls)] for arr = [mus, sigs, deltas]]
        params[key][u, :] = [muhat; sighat; deltahat] #store parameters
    end
end

#write to file
@save "$datadir/guided_lognormal_params_delta$wrapstr.bson" params #mu, sigma, delta

