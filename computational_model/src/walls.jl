### functions that are shared across environments ###

using Flux, Statistics, Random, Distributions, StatsFuns, Zygote, PyPlot

struct WallState
    reward_location::Array{Float32}
    wall_loc::Array{Int32}
    time::Array{Float32}
end

function WallState(; reward_location, wall_loc, time = zeros(1))
    return WallState(reward_location, wall_loc, time)
end


function state_ind_from_state(Larena, state)
    #input is 2 x batch
    #output is (batch, )
    return Larena * (state[1, :] .- 1) + state[2, :]
end

function onehot_from_loc(Larena, loc)
    #input: (batch, )
    #output: Nstates x batch
    Nstates = Larena^2
    batch = length(loc)
    shot = zeros(Nstates, batch)
    for b in 1:batch
        shot[loc[b], b] = 1
    end
    return shot # don't take gradients of this
end
Zygote.@nograd(onehot_from_loc)

function onehot_from_state(Larena, state)
    #input: 2 x batch
    #output: Nstates x batch
    state_ind = state_ind_from_state(Larena, state) # (batch,)
    return onehot_from_loc(Larena, state_ind) # don't take gradients of this
end
Zygote.@nograd(onehot_from_state)

function state_from_loc(Larena, loc)
    #input: 1 x batch
    #output: 2 x batch
    return [(loc .- 1) .÷ Larena .+ 1; (loc .- 1) .% Larena .+ 1]
end

function state_from_onehot(Larena, shot)
    #inpute: Nstates x batch
    #output: 2 x batch
    loc = [sortperm(-shot[:, b])[1] for b in 1:size(shot, 2)]
    loc = reduce(hcat, loc)
    return state_from_loc(Larena, loc)
end

function get_wall_input(state, wall_loc)
    #state is 2xB
    #wall_loc is Nstates x 4 x B (4 is right/left/up/down)
    input = [wall_loc[:, 1, :]; wall_loc[:, 3, :]] #all horizontal and all vertical walls
    return input # don't take gradients of this
end

function gen_input(
    world_state, ahot, rew, ed, model_properties
)
    batch = size(rew, 2)
    newstate = world_state.agent_state
    wall_loc = world_state.environment_state.wall_loc
    Naction = ed.Naction
    Nstates = ed.Nstates
    shot = onehot_from_state(ed.Larena, newstate) #one-hot encoding (Nstates x batch)
    wall_input = get_wall_input(newstate, wall_loc) #get input about walls
    Nwall_in = size(wall_input, 1)
    Nin = model_properties.Nin
    plan_input = world_state.planning_state.plan_input
    Nplan_in = size(plan_input, 1)

    ### speed this up ###
    x = zeros(Nin, batch)
    x[1:Naction, :] = ahot
    x[Naction + 1, :] = rew[:]
    x[Naction + 2, :] = world_state.environment_state.time / 50f0 #smaller time input in [0,1]
    x[(Naction + 3):(Naction + 2 + Nstates), :] = shot
    x[(Naction + 2 + Nstates + 1):(Naction + 2 + Nstates + Nwall_in), :] = wall_input

    if length(plan_input) > 0 #set planning input
        x[(Naction + 2 + Nstates + Nwall_in + 1):(Naction + 2 + Nstates + Nwall_in + Nplan_in), :] = world_state.planning_state.plan_input
    end

    return Float32.(x)
end


function get_rew_locs(reward_location)
    return [argmax(reward_location[:, i]) for i in 1:size(reward_location, 2)]
end
Zygote.@nograd get_rew_locs #don't take gradients of this
