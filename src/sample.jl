using StatsBase

#Take nsamples bitstrings from a 2D open boundary tensornetwork using boundary MPS with relevant ranks
function StatsBase.sample(
    ψ::ITensorNetwork,
    nsamples::Int64;
    left_message_rank::Int64=maxlinkdim(ψ),
    right_message_rank::Int64,
    boundary_mps_kwargs=get_global_boundarymps_update_kwargs(),
    bp_update_kwargs=get_global_bp_update_kwargs(),
    # set_bp_norm_to_one = true,
    # transform_to_symmetric_gauge = true, # TODO: do we even want control over this?
    kwargs...,
)
    ψIψ_bpc = build_bp_cache(ψ; bp_update_kwargs...)
    # if transform_to_symmetric_gauge
    #     ψ, ψIψ_bpc = symmetric_gauge(ψ; (cache!)=Ref(ψIψ_bpc), update_cache=false)
    # end
    # if set_bp_norm_to_one
    #     ψ, ψIψ_bpc = normalize(ψ, ψIψ_bpc; update_cache=false)
    # end

    ψ, ψIψ_bpc = symmetric_gauge(ψ)
    ψ, ψIψ_bpc = normalize(ψ, ψIψ_bpc; update_cache=false)

    right_MPScache = BoundaryMPSCache(ψIψ_bpc; message_rank=right_message_rank)
    sorted_partitions = sort(ITensorNetworks.partitions(right_MPScache))
    seq = [
        sorted_partitions[i] => sorted_partitions[i-1] for
        i = length(sorted_partitions):-1:2
    ]
    right_MPScache = updatecache(right_MPScache; boundary_mps_kwargs...) # update(Algorithm("orthogonal"), right_MPScache, seq; message_update_kwargs...)

    left_MPScache = BoundaryMPSCache(ψ; message_rank=left_message_rank)

    #Generate the bit_strings moving left to right through the network
    bit_strings = []
    for j = 1:nsamples
        p_over_q, bit_string = get_one_sample(
            right_MPScache, left_MPScache, sorted_partitions; boundary_mps_kwargs, kwargs...)
        push!(bit_strings, ((p_over_q), bit_string))
    end
    #norm = sum(first.(bit_strings)) / length(bit_strings)
    #bit_strings =
    #    [((p_over_q) / norm, bit_string) for (p_over_q, bit_string) in bit_strings]
    return bit_strings
end


function get_one_sample(
    right_MPScache::BoundaryMPSCache,
    left_MPScache::BoundaryMPSCache,
    sorted_partitions;
    boundary_mps_kwargs=get_global_boundarymps_update_kwargs(),
    kwargs...
)

    left_message_update_kwargs = (; boundary_mps_kwargs[:message_update_kwargs]..., normalize=false)

    right_MPScache = copy(right_MPScache)
    # left_incoming_message = nothing

    bit_string = Dictionary{keytype(vertices(left_MPScache)),Int}()
    p_over_q = nothing
    for (i, partition) in enumerate(sorted_partitions)

        right_MPScache, p_over_q, bit_string, =
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

    end

    return p_over_q, bit_string
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
        ρ = contract(environment(bp_cache(ψIψ), [(v, "operator")]); sequence="automatic")
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
        ψIψ = rem_vertex(ψIψ, (v, "operator"))
        ψIψ = update_factor(ψIψ, (v, "ket"), ψv)
        ψIψ = update_factor(ψIψ, (v, "bra"), dag(prime(ψv)))
        prev_v = v
    end

    return ψIψ, first(traces), bit_string
end
