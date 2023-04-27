#in this script, we estimate the space of environments spanned by our task set

#load some stuff
include("anal_utils.jl")
using ToPlanOrNotToPlan

println("estimating the total number of possible tasks")

#construct environment
model_properties, wall_environment, model_eval = build_environment(Larena, N, 50, Lplan = Lplan)

batch = 50000 #number of environments to create
Nstates = Larena^2 #number of unique states (and therefore potential reward locations)
all_Npairs, all_Nids = [], [] #total pairwise comparisons and total identical comparisons
Nseeds = 10 #repeat 10 times for uncertainty quantification
for seed = 1:Nseeds
    Random.seed!(seed) #set random seed for reproducibility
    #create environments
    world_state, agent_input = wall_environment.initialize(zeros(2), zeros(2), batch, model_properties)
    Ws = world_state.environment_state.wall_loc #all the wall locations

    Npairs, Nid = 0, 0 #start from zero comparisons
    for b1 = 1:batch #for each environment
        if b1 % 10000 == 0 println("seed $seed of $Nseeds, environment $b1: $Npairs pairwise comparisons, $Nid identical") end
        for b2 = b1+1:batch #for each different environment
            Npairs += 1 #one more pairwise comparison
            Nid += Int(Ws[:, :, b1] == Ws[:, :, b2]) #are these two mazes identical?
        end
    end

    frac_id = Nid/Npairs #fraction of identical wall layouts
    println("fraction identical: ", frac_id) #inverse of the number of wall layouts
    println("effective task space: ", Nstates/frac_id) #16 rew locations * 1/f wall layouts
    push!(all_Npairs, Npairs); push!(all_Nids, Nid)
end

#save result for future reference
result = Dict("Npairs" => all_Npairs, "Nids" => all_Nids)
@save "$datadir/estimate_num_mazes.bson" result

task_spaces = Nstates * all_Npairs ./ all_Nids #effective task space for each seed
num_mazes = mean(task_spaces) #mean
err = std(task_spaces)/sqrt(length(task_spaces)) #standard error
println("effective task space: ", num_mazes, " sem: ", err) #16 rew locations
