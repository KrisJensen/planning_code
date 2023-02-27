include("anal_utils.jl")
using Revise
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics, Zygote
using BSON: @save, @load

network, opt, store, hps, policy, prediction = recover_model("../models/maze/N100_T50_seed61_Lplan8_1000", modular = true)
model_properties, wall_environment, model_eval = build_environment(
    hps["Larena"], hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = true
)
m = ModularModel(model_properties, network, policy, prediction, forward_modular)

# run a handful of steps
batch = 50000; Ntrain = 1000*40*200; Nstates = 16
all_Npairs, all_Nids = [], []
for seed = 1:10
    Random.seed!(seed)
    world_state, agent_input = wall_environment.initialize(
                                                        zeros(2), zeros(2), batch, m.model_properties
                                                    )

    Ws = world_state.environment_state.wall_loc

    Npairs, Nid = 0, 0
    for b1 = 1:batch
        if b1 % 1000 == 0 println(b1, " ", Npairs, " ", Nid) end
        for b2 = b1+1:batch
            Npairs += 1
            Nid += Int(Ws[:, :, b1] == Ws[:, :, b2])
        end
    end

    frac_id = Nid/Npairs
    println("fraction identical: ", frac_id)
    println("effective task space: ", Nstates/frac_id) #16 rew locations
    println("number of training mazes: ", Ntrain)
    println("expected coverage: ", Ntrain * frac_id/Nstates)
    push!(all_Npairs, Npairs); push!(all_Nids, Nid)
end

result = Dict("Npairs" => all_Npairs, "Nids" => all_Nids)
@save "$datadir/estimate_num_mazes.bson" result

num_mazes = Nstates*mean(all_Npairs ./ all_Nids); err = Nstates*std(all_Npairs ./ all_Nids)/sqrt(length(all_Nids))
println("effective task space: ", num_mazes, " err: ", err) #16 rew locations
println("number of training mazes: ", Ntrain)
println("expected coverage: ", Ntrain / num_mazes)

### just check that this is a correct estimate###

batch2 = 3000
println(batch2*(batch2-1)/2)
for N = [100, 1000, 10000, 100000, 1000000, 10000000]
    nums = rand(1:N, batch2)
    Npairs2, Nid2 = 0, 0
    for b1 = 1:batch2
        for b2 = b1+1:batch2
            Npairs2 += 1
            Nid2 += Int(nums[b1] == nums[b2])
        end
    end
    println(log10(N), " effective task space: ", Npairs2/Nid2, " ", Nid2)
end


N = 100000; batch2 = 3800
ests, Nids = [], []
for i = 1:100
    println(i)
    nums = rand(1:N, batch2)
    Npairs2, Nid2 = 0, 0
    for b1 = 1:batch2
        for b2 = b1+1:batch2
            Npairs2 += 1
            Nid2 += Int(nums[b1] == nums[b2])
        end
    end
    #println(Nid2)
    push!(ests, Npairs2/Nid2); push!(Nids, Nid2)
end
println(mean(ests), " ", std(ests), " ", std(ests)/sqrt(length(ests)))

