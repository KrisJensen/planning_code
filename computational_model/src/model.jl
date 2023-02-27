using Zygote, Flux

struct ModelProperties
    Nout::Int
    Nhidden::Int
    Nin::Int
    Lplan::Int
    greedy_actions::Bool
    no_planning::Any #if true, never stand still
end

struct ModularModel
    model_properties::ModelProperties
    network::Any
    policy::Any
    prediction::Any
    forward::Function
end

function ModelProperties(; Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning = false)
    return ModelProperties(Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning)
end

function Modular_model(mp::ModelProperties, Naction::Int; Nstates = nothing, neighbor = false)
    # define our model!
    network = Chain(GRU(mp.Nin, mp.Nhidden))
    policy = Chain(Dense(mp.Nhidden, Naction+1)) #policy and value function
    Npred_out = mp.Nout - Naction - 1
    prediction = Chain(Dense(mp.Nhidden+Naction, Npred_out, relu), Dense(Npred_out, Npred_out))
    return ModularModel(mp, network, policy, prediction, forward_modular)
end

function build_model(mp::ModelProperties, Naction::Int)
    return Modular_model(mp, Naction)
end

function create_model_name(
    Nhidden::Int,
    T::Int,
    seed,
    Lplan::Int,
    prefix = ""
)
    #define some useful model name
    mod_name =
        prefix*
        "_N$Nhidden" *
        "_T$T" *
        "_seed$seed" *
        "_Lplan$Lplan"

    return mod_name
end
