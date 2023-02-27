## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics, Zygote
using BSON: @save

#planner_type = "model"
loss_hp = LossHyperparameters(;βp=0.5f0,βv=0.05f0,βe=0.05f0,βa=0.0f0,βr=1.0f0,γ=1.0f0,bootstrap=500,
                predict=true,epsilon = 0.2,Naction = 5,Larena = 4,prior_type = "uniform")
loss_hp = LossHyperparameters(;βp=0.0f0,βv=0.00f0,βe=0.05f0,βa=0.0f0,βr=1.0f0,γ=1.0f0,bootstrap=500,
                predict=true,epsilon = 0.2,Naction = 5,Larena = 4,prior_type = "uniform")
#loss_hp = LossHyperparameters(;βp=0.5f0,βv=0.05f0,βe=0.00f0,βa=0.0f0,βr=0.0f0,γ=1.0f0,bootstrap=500,
#                predict=true,epsilon = 0.2,Naction = 5,Larena = 4,prior_type = "uniform")

arena = build_arena(4)
epochs = [100;200;500;1000;1900]
planners = ["model"; "model_ctrl"; "model_ctrl2"]
noplans = [""; ""; "_noplan"]
seeds = 61:65
batch_size = 300
prefix = ""

meanrews = zeros(length(planners), length(epochs), length(seeds));
gradvars = zeros(length(planners), length(epochs), length(seeds));
gradmeans = zeros(length(planners), length(epochs), length(seeds));
all_gs = zeros(length(planners), length(epochs), length(seeds), 100*300, batch_size);

for (iplanner, planner) = enumerate(planners)
noplan = noplans[iplanner]
for (iepoch, epoch) = enumerate(epochs)
for (iseed, seed) = enumerate(seeds)

load_planner = planner; #load_planner = "model_ctrl"; noplan = ""
network, opt, store, hps, policy, prediction = recover_model("../models/maze/$(prefix)L4_N100_T50_seed$(seed)_planning_$(load_planner)_Lplan8$(noplan)_$epoch", modular = true)
model_properties, wall_environment, model_eval = build_environment(
    arena, hps["Nhidden"], hps["T"], attention = false, Lplan = hps["Lplan"], task="maze",
    greedy_actions = false, planner_type = planner, no_planning = (iplanner == 3), trap = false
);
m = ModularModel(model_properties, network, policy, prediction, forward_modular)
prms = Params(Flux.params(m.network, m.policy, m.prediction))

L, ys, rews, as, world_states, hs = run_episode(
        m, wall_environment, loss_hp; batch=5*batch_size, ponder = 0, calc_loss = false
    )
meanrew = sum(rews) / size(rews, 1)
meanrews[iplanner, iepoch, iseed] = meanrew

closure() = model_loss(m, wall_environment, loss_hp, 1, ponder = 0)
gs = reduce(hcat, [gradient(closure, prms).grads[prms[2]][:] for b = 1:batch_size]) #recurrent weights
gvar = mean(var(gs, dims = 2)) #compute variance of the gradient, then compute average across units
gradvars[iplanner, iepoch, iseed] = gvar
gmean = sqrt.(mean( mean(gs, dims = 2).^2 )) #compute mean gradient, then compute magnitude of this vector
gradmeans[iplanner, iepoch, iseed] = gmean
all_gs[iplanner, iepoch, iseed, :, :] = gs

println("$planner $epoch $seed: $meanrew, $gvar, $gmean")

end
end
end

rels = zeros(length(planners), length(epochs), length(seeds))
for iplanner = 1:length(planners)
    for iepoch = 1:length(epochs)
        for iseed = 1:length(seeds)
            gs = all_gs[iplanner, iepoch, iseed, :, :]
            gms = mean(gs, dims = 2)
            deltas = gs .- gms
            dists = sqrt.(sum(deltas.^2, dims = 1)[:]) #sqrt of eigenvalues of covariance matrix
            mmag = sqrt(sum(gms.^2))
            rels[iplanner, iepoch, iseed] = mean(dists)/mmag

            ### adam estimator ###
            gvs = sqrt.(mean(gs.^2, dims = 2))
            #rels[iplanner, iepoch, iseed] = sqrt( sum( (gms[gvs .> 0] ./ gvs[gvs .> 0]).^2 ) ) #magnitude of the adam update

        end
    end
end


mr, sr = mean(meanrews, dims = 3)[:, :, 1], std(meanrews, dims = 3)[:, :, 1]/sqrt(length(seeds))
mgv, sgv = mean(gradvars, dims = 3)[:, :, 1], std(gradvars, dims = 3)[:, :, 1]/sqrt(length(seeds))
mgm, sgm = mean(gradmeans, dims = 3)[:, :, 1], std(gradmeans, dims = 3)[:, :, 1]/sqrt(length(seeds))

#rels = gradvars ./ (gradmeans.^2)
mg, sg = mean(rels, dims = 3)[:, :, 1], std(rels, dims = 3)[:, :, 1]/sqrt(length(seeds))

#mg, sg = mgv, sgv

figure()
for i = 1:3
    if i in 1:2 inds = 1:size(mr, 2) else inds = 1:size(mr, 2)-1 end
    errorbar(mr[i,inds], mg[i,inds], xerr=sr[i,inds], yerr=sg[i,inds], ls = "-")
end
xlabel(L"\mathbb{E}[R]")
ylabel("var/mean")
savefig("./grad_var.png", bbox_inches = "tight")
close()

### note that the raw mean IS important because this is only one factor in our loss ###
figure()
for i = 1:3
    if i in 1:2 inds = 1:size(mr, 2) else inds = 1:size(mr, 2)-1 end
    errorbar(mr[i,inds], mgm[i,inds], xerr=sr[i,inds], yerr=sgm[i,inds], ls = "-")
end
xlabel(L"\mathbb{E}[R]")
ylabel(L"| \mathbb{E}_{\tau}[\nabla_{\bf \theta} \mathcal{L}] | ")
savefig("./grad_mean.png", bbox_inches = "tight")
close()

