function norm_sqr(tns::Union{TensorNetworkState, BeliefPropagationCache}; alg = nothing, kwargs...)
    norm_sqr(Algorithm(alg), tns; kwargs...)
end

function norm_sqr(alg::Algorithm"exact", ψ::TensorNetworkState;
    contraction_sequence_kwargs=(; alg="einexpr", optimizer=Greedy()))
    ψIψ_tensors = norm_factors(ψ, collect(vertices(ψ)))
    denom_seq = contraction_sequence(ψIψ_tensors; contraction_sequence_kwargs...)
    return contract(ψIψ_tensors; sequence=denom_seq)[]
end

function norm_sqr(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, bp_cache::BeliefPropagationCache; max_configuration_size = nothing)
    tn = network(bp_cache)
    z = alg == Algorithm("bp") ? partitionfunction(bp_cache) : loopcorrected_partitionfunction(bp_cache, max_configuration_size)
    tn isa TensorNetworkState && return z
    tn isa ITensorNetwork && return z*z
    error("Unrecognized network type inside the BP cache")
end

function norm_sqr(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, ψ::TensorNetworkState; bp_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    return norm_sqr(alg, ψ_bpc; kwargs...)
end

LinearAlgebra.norm(alg::Algorithm, ψ::Union{TensorNetworkState, BeliefPropagationCache}; kwargs...) = sqrt(norm_sqr(alg, ψ; kwargs...))
LinearAlgebra.norm(ψ::Union{TensorNetworkState, BeliefPropagationCache}; kwargs...) = sqrt(norm_sqr(ψ; kwargs...))