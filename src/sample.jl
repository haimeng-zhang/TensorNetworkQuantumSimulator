using StatsBase

#Take nsamples bitstrings from a 2D open boundary tensornetwork using boundary MPS with relevant ranks
function _sample(
    ψ::ITensorNetwork,
    nsamples::Int64;
    left_message_rank::Int64,
    right_message_rank::Int64,
    boundary_mps_kwargs=get_global_boundarymps_update_kwargs(),
    compute_independent_logp = false,
    kwargs...,
)
    ψ, ψIψ_bpc = symmetric_gauge(ψ)
    ψ, ψIψ_bpc = normalize(ψ, ψIψ_bpc; update_cache=false)

    right_MPScache = BoundaryMPSCache(ψIψ_bpc; message_rank=right_message_rank)
    sorted_partitions = sort(ITensorNetworks.partitions(right_MPScache))
    seq = [
        sorted_partitions[i] => sorted_partitions[i-1] for
        i = length(sorted_partitions):-1:2
    ]
    right_MPScache = updatecache(right_MPScache; boundary_mps_kwargs...)

    left_MPScache = BoundaryMPSCache(ψ; message_rank=left_message_rank)

    #Generate the bit_strings moving left to right through the network
    bit_strings = []
    for j = 1:nsamples
        p_over_q_approx, p_over_q_exact, bit_string = get_one_sample(
            right_MPScache, left_MPScache, sorted_partitions; compute_independent_logp, boundary_mps_kwargs, kwargs...)
        push!(bit_strings, ((p_over_q_approx, p_over_q_exact), bit_string))
    end
    #norm = sum(first.(bit_strings)) / length(bit_strings)
    #bit_strings =
    #    [((p_over_q) / norm, bit_string) for (p_over_q, bit_string) in bit_strings]
    return bit_strings
end

function StatsBase.sample(ψ::ITensorNetwork, nsamples::Int64; kwargs...)
    bitstrings = _sample(ψ::ITensorNetwork, nsamples::Int64; kwargs...)
    return last.(bitstrings)
end

function direct_importance_sample(ψ::ITensorNetwork, nsamples::Int64; left_message_rank = maxlinkdim(ψ), kwargs...)
    bitstrings = _sample(ψ::ITensorNetwork, nsamples::Int64; left_message_rank, kwargs...)
    return [(scalars[1], bitstring) for (scalars, bitstring) in bitstrings]
end

function indirect_importance_sample(ψ::ITensorNetwork, nsamples::Int64; kwargs...)
    bitstrings = _sample(ψ::ITensorNetwork, nsamples::Int64; compute_independent_logp = true, kwargs...)
    return [(scalars[2], bitstring) for (scalars, bitstring) in bitstrings]
end

function get_one_sample(
    right_MPScache::BoundaryMPSCache,
    left_MPScache::BoundaryMPSCache,
    sorted_partitions;
    boundary_mps_kwargs=get_global_boundarymps_update_kwargs(),
    compute_independent_logp = false,
    kwargs...
)

    left_message_update_kwargs = (; boundary_mps_kwargs[:message_update_kwargs]..., normalize=false)

    right_MPScache = copy(right_MPScache)

    bit_string = Dictionary{keytype(vertices(left_MPScache)),Int}()
    p_over_q_approx = nothing
    for (i, partition) in enumerate(sorted_partitions)

        right_MPScache, p_over_q_approx, bit_string, =
            sample_partition(right_MPScache, partition, bit_string; kwargs...)
        vs = planargraph_vertices(right_MPScache, partition)

        left_MPScache = update_factors(
            left_MPScache,
            Dict(zip(vs, [only(factors(right_MPScache, [(v, "ket")])) for v in vs])),
        )


        if i < length(sorted_partitions)
            next_partition = sorted_partitions[i+1]

            ms = messages(right_MPScache)

            left_MPScache = update(
                Algorithm("orthogonal"),
                left_MPScache,
                partition => next_partition;
                left_message_update_kwargs...,
            )

            pes = planargraph_sorted_partitionedges(right_MPScache, partition => next_partition)

            for pe in pes
                m = only(message(left_MPScache, pe))
                set!(ms, pe, [m, dag(prime(m))])
            end
        end

        delete_partition_messages!(right_MPScache, partition)
        if i != 1 && i != length(sorted_partitions) 
            delete_partition_messages!(left_MPScache, partition)
            delete_partitionpair_messages!(left_MPScache, sorted_partitions[i-1] => sorted_partitions[i])
            if i > 2
                delete_partitionpair_messages!(right_MPScache, sorted_partitions[i-2] => sorted_partitions[i-1])
            end
        end

    end

    !compute_independent_logp && return p_over_q_approx, nothing, bit_string

    ψproj = tensornetwork(left_MPScache)
    left_MPScache = BoundaryMPSCache(ψproj; message_rank = maxlinkdim(ψproj))
    left_MPScache = updatecache(left_MPScache)
    p_over_q_exact = scalar(left_MPScache)
    p_over_q_exact *= conj(p_over_q_exact)

    return p_over_q_approx, p_over_q_exact, bit_string
end


#Sample along the column/ row specified by pv with the left incoming MPS message input and the right extractable from the cache
function sample_partition(
    ψIψ::BoundaryMPSCache,
    partition,
    bit_string::Dictionary,
    kwargs...,
)
    vs = sort(planargraph_vertices(ψIψ, partition))
    prev_v, traces = nothing, []
    for v in vs
        ψIψ =
            !isnothing(prev_v) ? partition_update(ψIψ, [prev_v], [v]) :
            partition_update(ψIψ, [v])
        env = environment(bp_cache(ψIψ), [(v, "operator")])
        seq = contraction_sequence(env; alg = "optimal")
        ρ = contract(env; sequence=seq)
        ρ_tr = tr(ρ)
        push!(traces, ρ_tr)
        ρ /= ρ_tr
        # the usual case of single-site
        config = StatsBase.sample(1:length(diag(ρ)), Weights(real.(diag(ρ))))
        # config is 1 or 2, but we want 0 or 1 for the sample itself
        set!(bit_string, v, config - 1)
        s_ind = only(filter(i -> plev(i) == 0, inds(ρ)))
        P = onehot(s_ind => config)
        q = diag(ρ)[config]
        ψv = only(factors(ψIψ, [(v, "ket")])) / sqrt(q)
        ψv = P * ψv
        ψIψ = update_factor(ψIψ, (v, "operator"), ITensor(one(Bool)))
        ψIψ = update_factor(ψIψ, (v, "ket"), ψv)
        ψIψ = update_factor(ψIψ, (v, "bra"), dag(prime(ψv)))
        prev_v = v
    end

    return ψIψ, first(traces), bit_string
end
