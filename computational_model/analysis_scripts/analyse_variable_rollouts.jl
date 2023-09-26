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

prefix = "variable_"
seeds = 61:65
N, Lplan = 100, 8

println("running N=$N, L=$Lplan")
# correlation with human RT ####
repeat_human_actions(;seeds, N, Lplan, epoch, prefix = prefix, model_prefix = prefix)

# change in performance with replay number ###
run_perf_by_rollout_number(;seeds, N, Lplan, epoch, prefix = prefix, model_prefix = prefix)

# change in policy after successful/unsuccessful replay ####
run_causal_rollouts(;seeds, N, Lplan, epoch, prefix = prefix, model_prefix = prefix)

