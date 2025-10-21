
"""
    apply_gates(circuit::Vector, ψ::Union{TensorNetworkState, BeliefPropagationCache}; bp_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)
    Apply a sequence of gates to a `TensorNetworkState` or a `BeliefPropagationCache`` wrapping a `TensorNetworkState`` using Belief Propagation to update the environment.
    # Arguments
    - `circuit::Vector`: A vector of tuples where each tuple contains a gate (as an `ITensor`) and the vertices it acts on.
    - `ψ::TensorNetworkState`: The tensor network state to which the gates will be applied.
    - `bp_update_kwargs`: Keyword arguments for updating the Belief Propagation cache (reasonable default is set).
    - `kwargs...`: Additional keyword arguments for gate application and BP updates.
    # Returns
    - A tuple containing the updated `TensorNetworkState` or `BeliefPropagationCache` and a vector of truncation errors for each gate application.
end
"""
function apply_gates(
    circuit::Vector,
    ψ::TensorNetworkState;
    bp_update_kwargs = default_bp_update_kwargs(ψ),
    kwargs...,
)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

function apply_gates(
    circuit::Vector,
    ψ_bpc::BeliefPropagationCache;
    kwargs...,
)
    gate_vertices = [_tovec(gate[2]) for gate in circuit]
    circuit = toitensor(circuit, siteinds(network(ψ_bpc)))
    circuit = [adapt(ComplexF32, gate) for gate in circuit]
    circuit = [adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate) for gate in circuit]
    return apply_gates(circuit, ψ_bpc; gate_vertices, kwargs...)
end

function apply_gates(
    circuit::Vector{<:ITensor},
    ψ_bpc::BeliefPropagationCache;
    gate_vertices::Vector = neighbor_vertices.((network(ψ_bpc), ), circuit),
    apply_kwargs = (; ),
    bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
    update_cache = true,
    verbose = false,
)
    ψ_bpc = copy(ψ_bpc)

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_vertices = Set()
    truncation_errors = zeros((length(circuit)))

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        cache_update_required = length(gate_vertices[ii]) >= 2 && any(vert in affected_vertices for vert in gate_vertices[ii])

        # update the BP cache
        if update_cache && cache_update_required
            if verbose
                println("Updating BP cache")
            end

            t = @timed ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

            affected_vertices = Set()
            if verbose
                println("Done in $(t.time) secs")
            end

        end

        # actually apply the gate
        t = @timed ψ_bpc, truncation_errors[ii] = apply_gate!(gate, ψ_bpc; v⃗ = gate_vertices[ii], apply_kwargs)
        affected_vertices = union(affected_vertices, Set(gate_vertices[ii]))
    end

    if update_cache
        ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    end

    return ψ_bpc, truncation_errors
end

#Apply function for a single gate
function apply_gate!(
    gate::ITensor,
    ψ_bpc::BeliefPropagationCache;
    v⃗ = ITensorNetworks.neighbor_vertices(ψ_bpc, gate),
    apply_kwargs = _default_apply_kwargs,
)
    envs = length(v⃗) == 1 ? nothing : incoming_messages(ψ_bpc,v⃗)

    updated_tensors, s_values, err = simple_update(gate, network(ψ_bpc), v⃗; envs, apply_kwargs...)

    if length(v⃗) == 2
        v1, v2 = v⃗
        e = NamedEdge(v1 => v2)
        ind2 = commonind(s_values, first(updated_tensors))
        δuv = dag(copy(s_values))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        s_values = denseblocks(s_values) * denseblocks(δuv)
        setmessage!(ψ_bpc, e, dag(s_values))
        setmessage!(ψ_bpc, reverse(e), s_values)
    end

    for (i, v) in enumerate(v⃗)
        setindex_preserve_graph!(ψ_bpc, updated_tensors[i], v)
    end

    return ψ_bpc, err
end

function simple_update(
    o::ITensor, ψ, v⃗; envs, normalize_tensors = true, apply_kwargs...
  )

    if length(v⃗) == 1
        updated_tensors = ITensor[ITensors.apply(o, ψ[first(v⃗)])]
        s_values, err = nothing, 0
    else
        cutoff = 10 * eps(real(scalartype(ψ[v⃗[1]])))
        envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))

        sqrt_inv_sqrt_envs_v1 = pseudo_sqrt_inv_sqrt.(envs_v1)
        sqrt_inv_sqrt_envs_v2 = pseudo_sqrt_inv_sqrt.(envs_v2)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        ψᵥ₁ = contract([ψ[v⃗[1]]; sqrt_envs_v1])
        ψᵥ₂ = contract([ψ[v⃗[2]]; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ[v⃗[1]], o)
        sᵥ₂ = commoninds(ψ[v⃗[2]], o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
        e = v⃗[1] => v⃗[2]
        singular_values! = Ref(ITensor())
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR,
        unioninds(rᵥ₁, sᵥ₁);
        ortho="none",
        tags=edge_tag(e),
        singular_values!,
        apply_kwargs...,
        )
        err = spec.truncerr
        s_values = singular_values![]
        Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
        Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
        updated_tensors = [Qᵥ₁ * Rᵥ₁, Qᵥ₂ * Rᵥ₂]
    end

    if normalize_tensors
        updated_tensors = ITensor[ψᵥ / norm(ψᵥ) for ψᵥ in updated_tensors]
    end

    return noprime.(updated_tensors), s_values, err
end
