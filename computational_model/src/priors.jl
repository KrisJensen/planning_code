

function U_prior(state, Naction)
    #uniform prior
    Zygote.ignore() do
        batch = size(state, 2)
        return ones(Float32, Naction, batch) / Naction
    end
end

function prior_loss(agent_output, state, active, mod)
    #return -KL[q || p]
    #return -KL[q || p]
    act = Float32.(Flux.unsqueeze(active, 1))
    Naction = length(mod.policy[1].bias)-1
    logp = log.(U_prior(state, Naction)) #KL regularization with uniform prior
    logπ = agent_output[1:Naction, :]
    if mod.model_properties.no_planning
        logπ = logπ[1:Naction-1, :] .- Flux.logsumexp(logπ[1:Naction-1, :]; dims=1)
        logp = logp[1:Naction-1, :] .- Flux.logsumexp(logp[1:Naction-1, :]; dims=1)
    end
    logp = logp .* act
    logπ = logπ .* act
    lprior = sum(exp.(logπ) .* (logp - logπ )) #-KL
    return lprior
end
