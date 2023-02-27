using BSON: @load, @save

function recover_model(filename)

    #@load filename * "_opt.bson" opt
    opt = nothing
    @load filename * "_hps.bson" hps
    @load filename * "_progress.bson" store
    @load filename * "_mod.bson" network
    @load filename * "_policy.bson" policy
    @load filename * "_prediction.bson" prediction
    return network, opt, store, hps, policy, prediction
end

function save_model(m, store, opt, filename, environment, loss_hp; Lplan)
    model_properties = m.model_properties
    network = m.network
    hps = Dict(
        "Nhidden" => model_properties.Nhidden,
        "T" => environment.properties.dimensions.T,
        "Larena" => environment.dimensions.Larena,
        "Nin" => model_properties.Nin,
        "Nout" => model_properties.Nout,
        "GRUind" => ToPlanOrNotToPlan.GRUind,
        "βp" => loss_hp.βp,
        "βe" => loss_hp.βe,
        "βr" => loss_hp.βr,
        "Lplan" => Lplan,
    )
    @save filename * "_progress.bson" store
    @save filename * "_mod.bson" network
    @save filename * "_opt.bson" opt
    @save filename * "_hps.bson" hps

    if :policy in fieldnames(typeof(m))
        policy = m.policy
        @save filename * "_policy.bson" policy
    end
    if :prediction in fieldnames(typeof(m))
        prediction = m.prediction
        @save filename * "_prediction.bson" prediction
    end
    if :prediction_state in fieldnames(typeof(m))
        prediction_state = m.prediction_state
        @save filename * "_prediction_state.bson" prediction_state
    end
end
