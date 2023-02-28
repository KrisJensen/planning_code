## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics
using BSON: @save

loss_hp = LossHyperparameters(0, 0, 0, 0)

greedy_actions = true
seeds = 61:65
epoch = plan_epoch
results = Dict()
batch_size = 50000

for seed = seeds

    ##
    results[seed] = Dict()
    for plan = [false; true]
        Random.seed!(1)

        network, opt, store, hps, policy, prediction = recover_model("../models/maze/N100_T50_seed$(seed)_Lplan8_$epoch")
        Larena = hps["Larena"]
        model_properties, wall_environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions,no_planning = (~plan)
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)
        Nstates = Larena^2

        tic = time()
        L, ys, rews, as, world_states, hs = run_episode(
            m, wall_environment, loss_hp; batch=batch_size, calc_loss = false
        )
        states = reduce((a, b) -> cat(a, b, dims = 3), [ws.agent_state for ws = world_states])
        wall_loc, ps = world_states[1].environment_state.wall_loc, world_states[1].environment_state.reward_location
        Tmax = size(as, 2)
        rew_locs = reshape(ps, Nstates, batch_size, 1) .* ones(1, 1, Tmax) #for each time point
        println(seed, " ", plan)
        println(sum(rews .> 0.5) / batch_size, " ", time() - tic)
        println("planning fraction: ", sum(as .> 4.5) / sum(as .> 0.5))
        results[seed][plan] = rews
    end
end

@save "$(datadir)/performance_with_out_planning.bson" results



