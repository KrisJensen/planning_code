struct LossHyperparameters
    βv::Float32
    βe::Float32
    βp::Float32
    βr::Float32
end

function LossHyperparameters(; βv, βe, βp, βr)
    return LossHyperparameters(βv, βe, βp, βr)
end

