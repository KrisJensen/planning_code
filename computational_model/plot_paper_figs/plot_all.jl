#### main text figures ####
include("plot_utils.jl")

using Suppressor
@suppress_err begin

#Fig 2
println("\nplotting response time figure")
include("plot_fig_RTs.jl")

#Fig 3
println("\nplotting RNN behavior figure")
include("plot_fig_mechanism_behav.jl")

#Fig 4
println("\nplotting replay figure")
include("plot_fig_replays.jl")

#Fig 5
println("\nplotting PG figure")
include("plot_fig_mechanism_neural.jl")

#### supplementary figures ####

#Sfig 1
println("\nplotting supplementary Euclidean comparison")
include("plot_supp_human_euc_comparison.jl")

#Sfig 2
println("\nplotting supplementary human data")
include("plot_supp_human_summary.jl")

#Sfig 3
println("\nplotting supplementary learning analyses")
include("plot_supp_fig_network_size.jl")

#Sfig 4
println("\nplotting supplementary RT by step within trial")
include("plot_supp_RT_by_step.jl")

#Sfig 5
println("\nplotting supplementary hp sweep")
#include("plot_supp_hp_sweep.jl")

#Sfig 6
println("\nplotting supplementary value function analyses")
include("plot_supp_values.jl")

#Sfig 7
println("\nplotting supplementary internal model")
include("plot_supp_internal_model.jl")

#Sfig 8
println("\nplotting supplementary analyses with variable rollout durations")
include("plot_supp_variable.jl")

#Sfig 9
println("\nplotting supplementary exploration analyses")
include("plot_supp_exploration.jl")

#Sfig 13
println("\nplotting supplementary re-plan probabilities")
include("plot_supp_plan_probs.jl")



end

