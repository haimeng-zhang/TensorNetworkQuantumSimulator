default_truncate_alg(tns::TensorNetworkState) = nothing

function ITensors.truncate(bpc::BeliefPropagationCache; bp_update_kwargs = default_bp_update_kwargs(bpc), maxdim::Integer, cutoff::Number = nothing)
    bpc = copy(bpc)
    s = siteinds(network(bpc))
    apply_kwargs = (; maxdim, cutoff)
    for e in edges(bpc)
        g1, g2 = reduce(*, [ITensors.op("I", sv) for sv in s[src(e)]]), reduce(*, [ITensors.op("I", sv) for sv in s[dst(e)]])
        apply_gate!(g1*g2, bpc; v⃗ = [src(e), dst(e)], apply_kwargs)
        bpc = update(bpc; bp_update_kwargs...)
    end
    return bpc
end

function ITensors.truncate(alg::Algorithm"bp", tns::TensorNetworkState; kwargs...)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = truncate(bp_cache; kwargs...)
    return network(bp_cache)
end

function ITensors.truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)
    algorithm_check(tns, "truncate", alg)
    return truncate(Algorithm(alg), tns; kwargs...)
end

