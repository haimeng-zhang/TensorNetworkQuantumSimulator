function inner_algorithm_error()
    error("Algorithm choice not supported. Currently supported: bp, boundarymps, loopcorrections and exact.")
end

function inner_state_error()
    error("Network type inside the cache is not a BilinearForm.")
end

function ITensors.inner(ψ::TensorNetworkState, ϕ::TensorNetworkState; alg = nothing, kwargs...)
    inner(Algorithm(alg), ψ, ϕ; kwargs...)
end

function ITensors.inner(alg::Algorithm"exact", blf::BilinearForm;
    contraction_sequence_kwargs=(; alg="einexpr", optimizer=Greedy()))
    blf_tensors = bp_factors(blf, collect(vertices(ket(blf))))
    seq = contraction_sequence(blf_tensors; contraction_sequence_kwargs...)
    return contract(blf_tensors; sequence=seq)[]
end

function ITensors.inner(alg::Algorithm, cache::AbstractBeliefPropagationCache; max_configuration_size = nothing)
    tn = network(cache)
    if alg == Algorithm("bp") || alg == Algorithm("boundarymps")
        z = partitionfunction(cache)
    elseif alg == Algorithm("loopcorrections")
        z = loopcorrected_partitionfunction(cache, max_configuration_size)
    else
        return inner_algorithm_error()
    end

    tn isa BilinearForm && return z
    return inner_state_error()
end

function ITensors.inner(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, ψ::TensorNetworkState, ϕ::TensorNetworkState; cache_update_kwargs = (;), kwargs...)
    ψϕ_bpc = BeliefPropagationCache(BilinearForm(ψ, ϕ))
    ψϕ_bpc = update(ψϕ_bpc; cache_update_kwargs...)
    return inner(alg, ψϕ_bpc; kwargs...)
end

function ITensors.inner(alg::Algorithm"boundarymps", ψ::TensorNetworkState, ϕ::TensorNetworkState; mps_bond_dimension::Int, cache_update_kwargs = (; ), kwargs...)
    ψϕ_bmps = BoundaryMPSCache(BilinearForm(ψ, ϕ), mps_bond_dimension)
    maxiter = get(cache_update_kwargs, :maxiter,  default_bp_maxiter(ψϕ_bmps))
    cache_update_kwargs = (; cache_update_kwargs..., maxiter)
    ψϕ_bmps = update(ψϕ_bmps; cache_update_kwargs...)
    return inner(alg, ψϕ_bmps; kwargs...)
end

function ITensors.inner(alg::Algorithm"exact", ψ::TensorNetworkState, ϕ::TensorNetworkState)
    return inner(alg, BilinearForm(ψ, ϕ))
end

function ITensors.inner(alg::Algorithm, ψ::TensorNetworkState, ϕ::TensorNetworkState; kwargs...)
    return algorithm_error()
end
