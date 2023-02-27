struct PlanState
    plan_input::Array{Float32}
    plan_cache::Any
end

struct Planner
    Lplan::Int
    Nplan_in::Int
    Nplan_out::Int
    planning_time::Float32
    planning_cost::Float32
    planning_algorithm::Function
end

function none_planner(world_state,
                    ahot,
                    ep,
                    agent_output,
                    at_rew,
                    planner,
                    model,
                    h_rnn,
                    mp)
    batch = size(ahot, 2)
    xplan = zeros(Float32, 0, batch) #no input
    plan_inds = [] #no indices
    plan_cache = nothing #no cache
    planning_state = PlanState(xplan, plan_cache)

    return planning_state, plan_inds
end

function build_planner(Lplan, Larena; planning_time = 1f0, planning_cost = 0f0)
    Nstates = Larena^2

    if Lplan <= 0.5
        Nplan_in, Nplan_out = 0, 0
        planning_algorithm = none_planner
        initial_plan_state = (batch -> PlanState([], []))
    else
        Nplan_in = 4*Lplan+1 #action sequence and whether we ended at the reward location
        Nplan_out = Nstates #rew location
        planning_algorithm = model_planner
        planning_time = 0.3f0 #planning needs to be fairly cheap here
        initial_plan_state = (batch -> PlanState([], [])) #we don't use a cache
    end

    planner = Planner(Lplan, Nplan_in, Nplan_out, planning_time, planning_cost, planning_algorithm)
    return planner, initial_plan_state
end

