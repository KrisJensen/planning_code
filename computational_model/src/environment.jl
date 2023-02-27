struct EnvironmentDimensions
    Nstates::Int
    Nstate_rep::Int
    Naction::Int
    T::Int
    Larena::Int
end

struct Environment{T}
    initialize::Function
    step::Function
    dimensions::EnvironmentDimensions{T}
end

struct WorldState
    agent_state
    environment_state
    planning_state
end

function WorldState(; agent_state, environment_state, planning_state)
    return WorldState(agent_state, environment_state, planning_state)
end
