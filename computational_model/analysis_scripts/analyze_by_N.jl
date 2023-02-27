include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics
using BSON: @save, @load

loss_hp = LossHyperparameters(0, 0, 0, 0, 0, 0, 1000, true, 0f0, () -> ())
Save = true

greedy_actions = true
no_planning = false
seeds = 61:65
Nhiddens = 40:20:120
epochs = 0:100:2000

loaddir = "../models/maze/"
resdir, figdir = "./results/", "../figs/maze/model/"

meanrews, pfracs = [zeros(length(Nhiddens), length(seeds), length(epochs)) for _ = 1:2]
pairwise_dists = zeros(length(Nhiddens), length(seeds), length(epochs), 16, 16)

for (ihid, Nhidden) = enumerate(Nhiddens)
for (iseed, seed) = enumerate(seeds)
for (iepoch, epoch) = enumerate(epochs)

filename = "$loaddir/N$(Nhidden)_T50_seed$(seed)_Lplan8_$epoch"
@load filename * "_hps.bson" hps
@load filename * "_progress.bson" store
@load filename * "_mod.bson" network
@load filename * "_policy.bson" policy
@load filename * "_prediction.bson" prediction

Larena = hps["Larena"]
model_properties, wall_environment, model_eval = build_environment(
    Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
)
m = ModularModel(model_properties, network, policy, prediction, forward_modular)

Random.seed!(1)
batch_size = 5000
tic = time()
L, ys, rews, as, world_states, hs = run_episode(
    m, wall_environment, loss_hp; batch=batch_size, calc_loss = false
)
states = reduce((a, b) -> cat(a, b, dims = 3), [ws.agent_state for ws = world_states])
wall_loc, ps = world_states[1].environment_state.wall_loc, world_states[1].environment_state.reward_location
plan_frac = sum(as .== 5)/sum(as .> 0.5)
mean_rew = sum(rews .> 0.5) / batch_size

println(Nhidden, " ", seed, " ", epoch, " ", mean_rew, " ", plan_frac)

meanrews[ihid, iseed, iepoch] = mean_rew
pfracs[ihid, iseed, iepoch] = plan_frac

### compute representational similarity ###
Rmin, dt = 2, 2
keep_inds = findall(sum(rews .> 0.5; dims=2)[:] .>= Rmin)
all_hs, ps = hs[:, keep_inds, :], world_states[1].environment_state.reward_location
mean_states = calc_mean_rep(dt, ps, keep_inds, rews, all_hs; trialnum=2)
for i1 in 1:Larena^2
    for i2 in 1:Larena^2
        pairwise_dists[ihid, iseed, iepoch, i1, i2] = sqrt(
            sum((mean_states[:, i1] - mean_states[:, i2]) .^ 2)
        )
    end
end


end
end
end

res_dict = Dict("seeds" => seeds,
                "Nhiddens" => Nhiddens,
                "epochs" => epochs,
                "meanrews" => meanrews,
                "planfracs" => pfracs,
                "pairwise" => pairwise_dists)

@save resdir * "rew_and_plan_by_n_new.bson" res_dict


