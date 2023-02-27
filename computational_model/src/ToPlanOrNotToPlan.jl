module ToPlanOrNotToPlan

include("plotting.jl")
include("train.jl")
include("model.jl")
include("loss_hyperparameters.jl")
include("environment.jl")
include("a2c.jl")
include("initializations.jl")
include("walls.jl")
include("maze.jl")
include("walls_build.jl")
include("io.jl")
include("model_planner.jl")
include("planning.jl")
include("human_utils_maze.jl")
include("exports.jl")
include("priors.jl")
include("walls_baselines.jl")

rc("font", size = 16)
PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

end
