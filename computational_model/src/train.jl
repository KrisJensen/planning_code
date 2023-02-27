using Flux, Zygote

function gmap(f, prms, gss::Zygote.ADictOrGrads...)
    gsout = Zygote.Grads(IdDict{Any,Any}(), prms)
    return gmap!(f, gsout, gss...)
end

function gmap!(f, gsout::Zygote.Grads, gss::Zygote.ADictOrGrads...)
    for (ip, p) in enumerate(gsout.params)
        gsout[p] = f((_getformap(gs, gs.params[ip]) for gs in gss)...)
    end
    return gsout
end
function _getformap(gs, p)
    g = gs[p]
    return isnothing(g) ? fill!(similar(p), 0) : g
end