include("anal_utils.jl")
using BSON: @load
using BSON: @save
using ToPlanOrNotToPlan
include("repeat_human_actions.jl")
include("perf_by_plan_number.jl")
include("analyze_planning_causality.jl")

greedy_actions = true
prefix = "hp_sweep_"
epoch = plan_epoch
epoch = 1000
prior = ""

seeds = 51:55
sizes = [60;100;140]
Lplans = [4;8;12]
epoch = 1000
N = 60; Lplan = 4

for N = sizes
    for Lplan = Lplans
        println("running N=$N, L=$Lplan")
        #### correlation with human RT ####
        repeat_actions(;seeds, prefix, epoch, prior, greedy_actions, N, Lplan)

        #### change in performance with replay number ###
        run_perf_by_plan_number(;seeds, prefix, N, Lplan, epoch, greedy_actions)

        #### change in policy after successful/unsuccessful replay ####
        run_planning_causal(;seeds, prefix, N, Lplan, epoch, greedy_actions)

    end
end

