function LinearAlgebra.normalize(alg::Algorithm"bp", tns::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(tns))
    tns_bpc = BeliefPropagationCache(tns)
    tns_bpc = update(tns_bpc; cache_update_kwargs...)
    rescale!(tns_bpc)
    return network(tns_bpc)
end

function LinearAlgebra.normalize(tns::TensorNetworkState; alg = nothing, kwargs...)
    return normalize(Algorithm(alg), tns; kwargs...)
end