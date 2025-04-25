using StatsBase

#Take nsamples bitstrings from a 2D open boundary tensornetwork using boundary MPS with relevant ranks
function StatsBase.sample(
    ψ::ITensorNetwork,
    nsamples::Int64;
    left_message_rank::Int64 = maxlinkdim(ψ),
    right_message_rank::Int64,
    message_update_kwargs = (; niters = 40, tolerance = 1e-12),
    bp_update_kwargs = get_global_bp_update_kwargs(),
    set_bp_norm_to_one = true,
    transform_to_symmetric_gauge = true,
    kwargs...,
)
    ψIψ_bpc = build_bp_cache(ψ; bp_update_kwargs...)
    if transform_to_symmetric_gauge
        ψ, ψIψ_bpc = symmetric_gauge(ψ; cache! = Ref(ψIψ_bpc), update_cache = false)
    end
    if set_bp_norm_to_one
        ψ, ψIψ_bpc = normalize(ψ, ψIψ_bpc; update_cache = false)
    end
    ψIψ = BoundaryMPSCache(ψIψ_bpc; message_rank = right_message_rank)
    sorted_partitions = sort(ITensorNetworks.partitions(ψIψ))
    seq = [
        sorted_partitions[i] => sorted_partitions[i-1] for
        i = length(sorted_partitions):-1:2
    ]
    ψIψ = update(Algorithm("orthogonal"), ψIψ, seq; message_update_kwargs...)

    pψ = BoundaryMPSCache(ψ; message_rank = left_message_rank)
    left_message_update_kwargs = (; message_update_kwargs..., normalize = false)

    #Generate the bit_strings moving left to right through the network
    bit_strings = []
    for j = 1:nsamples
        ψIψ_j = copy(ψIψ)
        left_incoming_message = nothing
        bit_string = Dictionary{keytype(vertices(ψ)),Int}()
        p_over_q = nothing
        for (i, partition) in enumerate(sorted_partitions)
            next_partition =
                i != length(sorted_partitions) ? sorted_partitions[i+1] : nothing
            ψIψ_j, p_over_q, bit_string, =
                sample_partition(ψIψ_j, partition, bit_string; kwargs...)
            vs = planargraph_vertices(ψIψ, partition)
            pψ = update_factors(
                pψ,
                Dict(zip(vs, [only(factors(ψIψ_j, [(v, "ket")])) for v in vs])),
            )
            if !isnothing(next_partition)
                ms = messages(ψIψ_j)
                pψ = update(
                    Algorithm("orthogonal"),
                    pψ,
                    partition => next_partition;
                    left_message_update_kwargs...,
                )
                pes = planargraph_sorted_partitionedges(ψIψ_j, partition => next_partition)
                for pe in pes
                    m = only(message(pψ, pe))
                    set!(ms, pe, [m, dag(prime(m))])
                end
            end

        end
        push!(bit_strings, ((p_over_q), bit_string))
    end
    #norm = sum(first.(bit_strings)) / length(bit_strings)
    #bit_strings =
    #    [((p_over_q) / norm, bit_string) for (p_over_q, bit_string) in bit_strings]
    return bit_strings
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
        ρ = contract(environment(bp_cache(ψIψ), [(v, "operator")]); sequence = "automatic")
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
