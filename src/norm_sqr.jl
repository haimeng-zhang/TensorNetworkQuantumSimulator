function algorithm_error()
    error("Algorithm choice not supported. Currently supported: bp, boundarymps, loopcorrections and exact.")
end

function state_error()
    error("Network type inside is not a TensorNetworkState.")
end

function norm_sqr(tns::Union{TensorNetworkState, BeliefPropagationCache}; alg = nothing, kwargs...)
    norm_sqr(Algorithm(alg), tns; kwargs...)
end

function norm_sqr(alg::Algorithm"exact", ψ::TensorNetworkState;
    contraction_sequence_kwargs=(; alg="einexpr", optimizer=Greedy()))
    ψIψ_tensors = norm_factors(ψ, collect(vertices(ψ)))
    denom_seq = contraction_sequence(ψIψ_tensors; contraction_sequence_kwargs...)
    return contract(ψIψ_tensors; sequence=denom_seq)[]
end

function norm_sqr(alg::Algorithm, cache::AbstractBeliefPropagationCache; max_configuration_size = nothing)
    tn = network(cache)

    if alg == Algorithm("bp") || alg == Algorithm("boundarymps")
        z = partitionfunction(cache)
    elseif alg == Algorithm("loopcorrections")
        z = loopcorrected_partitionfunction(cache, max_configuration_size)
    else
       return algorithm_error()
    end

    tn isa TensorNetworkState && return z
    tn isa ITensorNetwork && return z*z
    return state_error()
end

function norm_sqr(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, ψ::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
    return norm_sqr(alg, ψ_bpc; kwargs...)
end

function norm_sqr(alg::Algorithm"boundarymps", ψ::TensorNetworkState; mps_bond_dimension::Int, cache_update_kwargs = default_bmps_update_kwargs(ψ), kwargs...)
    ψ_bmps = BoundaryMPSCache(ψ, mps_bond_dimension)
    maxiter = get(cache_update_kwargs, :maxiter,  default_bp_maxiter(ψ_bmps))
    cache_update_kwargs = (; cache_update_kwargs..., maxiter)
    ψ_bmps = update(ψ_bmps; cache_update_kwargs...)
    return norm_sqr(alg, ψ_bmps; kwargs...)
end

function norm_sqr(alg::Algorithm, ψ::TensorNetworkState; kwargs...)
    return algorithm_error()
end

LinearAlgebra.norm(alg::Algorithm, ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}; kwargs...) = sqrt(norm_sqr(alg, ψ; kwargs...))
LinearAlgebra.norm(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}; kwargs...) = sqrt(norm_sqr(ψ; kwargs...))