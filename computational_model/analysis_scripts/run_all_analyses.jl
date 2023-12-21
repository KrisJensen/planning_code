# in this script, we call all of the model analysis functions.
# this may take a while to run unless you have a very big computer

println("running all analyses")
tic = time()

include("analyse_human_data.jl")
include("calc_human_prior.jl")

include("analyse_rollout_timing.jl")
include("repeat_human_actions.jl")
include("perf_by_rollout_number.jl")
include("compare_perf_without_rollout.jl")
include("shuffle_rollout_times.jl")
include("behaviour_by_success.jl")
include("model_replay_analyses.jl")
include("rollout_as_pg.jl")
#include("analyse_by_N.jl") [need to re-train models or include pretrained models]
include("estimate_num_mazes.jl")

include("compare_maze_path_lengths.jl")
#include("analyse_hp_sweep.jl") [need to re-train models or include pretrained models]
include("eval_value_function.jl")
#include("quantify_internal_model.jl") [need to re-train models or include full training run in pretrained models]
#include("analyse_variable_rollouts.jl") [need to re-train models or include pretrained models]

println("\nFinished after ", (time()-tic)/60/60, " hours.")
