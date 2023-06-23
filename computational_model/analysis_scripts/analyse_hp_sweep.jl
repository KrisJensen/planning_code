#in this script, we repeat some key analyses with different network sizes and rollout lengths
#we do this to assess the robustness of our results

# load some scripts
include("anal_utils.jl")

global run_default_analyses = false # load functions without running analyses for default models
include("repeat_human_actions.jl")
include("perf_by_rollout_number.jl")
include("behaviour_by_success.jl")
global run_default_analyses = true # back to default

println("repeating analyses with different hyperparameters")

prefix = ""
seeds = 51:55 #use a separate set of seeds
sizes = [60;100;140] #model sizes to consider
Lplans = [4;8;12] #planning horizons to consider

for N = sizes #for each network size
    for Lplan = Lplans #for each planning horizon
        println("running N=$N, L=$Lplan")
        # correlation with human RT ####
        #repeat_human_actions(;seeds, N, Lplan, epoch, prefix = "hp_")

        # change in performance with replay number ###
        run_perf_by_rollout_number(;seeds, N, Lplan, epoch, prefix = "hp_")

        # change in policy after successful/unsuccessful replay ####
        run_causal_rollouts(;seeds, N, Lplan, epoch, prefix = "hp_")
    end
end

