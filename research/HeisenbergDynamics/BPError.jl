using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using NPZ

using ITensorMPS
using Statistics
using EinExprs

using ITensorNetworks: IndsNetwork, AbstractBeliefPropagationCache, BeliefPropagationCache, random_tensornetwork, incoming_messages, edge_tag,
    set_message!, message

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, inds, commoninds

using NamedGraphs.GraphsExtensions: add_edges, add_vertices, rem_vertices

using NamedGraphs.PartitionedGraphs: PartitionVertex, PartitionEdge, partitionedge, partitioned_graph, boundary_partitionedges

using ITensorNetworks: ITensorsExtensions

using Serialization

using EinExprs: Greedy

using Statistics

function bp_error(bpc::BeliefPropagationCache, path::Vector{<:PartitionEdge}, target_pe::PartitionEdge; no_vals = minimum((prod(dim.(linkinds(bpc, target_pe))), 2)))

    is_tree(partitioned_graph(bpc)) && return 0
    bpc = copy(bpc)
    rev_path = reverse(reverse.(path))
    x0, y0 = only(message(bpc, target_pe)), only(message(bpc, reverse(target_pe)))
    n = prod(dim.(linkinds(bpc, target_pe)))

    function message_update(bpc::BeliefPropagationCache, pe::PartitionEdge, pe_in::PartitionEdge; conj_elements = false)
        vertex = src(pe)
        other_incoming_ms = incoming_messages(bpc, vertex; ignore_edges=PartitionEdge[pe_in, reverse(pe)])
        incoming_m = only(message(bpc, pe_in))
        state = factors(bpc, vertex)

        ts = conj_elements ? ITensor[incoming_m; conj.(other_incoming_ms); conj.(state)] : ITensor[incoming_m; other_incoming_ms; state]
        seq  = contraction_sequence(ts; alg = "optimal")
        return contract(ts; sequence = seq)
    end

    function map(m::ITensor, flag)

        adjoint_map = flag == Val(true)
        _m = copy(m)
        prev_pe = !adjoint_map ? target_pe : reverse(target_pe)
        pth = !adjoint_map ? path : rev_path
        set_message!(bpc, prev_pe, ITensor[_m])
        for pe in pth
            _m = message_update(bpc, pe, prev_pe; conj_elements = adjoint_map)
            set_message!(bpc, pe, ITensor[_m])
            prev_pe = pe
        end
        final_pe = !adjoint_map ? target_pe : reverse(target_pe)
        _m = message_update(bpc, final_pe, prev_pe; conj_elements = adjoint_map)
        return _m
    end

    #vals, lvecs, rvecs, info = svdsolve(map, x0, no_vals, :LR)
    lvals, lvecs, info = eigsolve(m -> map(m, Val(false)), x0, no_vals, :LM; tol = 1e-14)
    rvals, rvecs, info = eigsolve(m -> map(m, Val(true)), x0, no_vals, :LM; tol = 1e-14)

    for (lval, rval) in zip(lvals, rvals)
        @assert (abs(lval) <= 1e-8 && abs(rval) <= 1e-8) || (abs(abs(lval) - abs(rval)) / abs(lval) <= 1e-6)
    end
    
    err = 1.0 - abs(lvals[1]) / sum(abs.(lvals))

    return err
end

function bp_error_exact(bpc::BeliefPropagationCache, path::Vector{<:PartitionEdge}, target_pe::PartitionEdge)

    is_tree(partitioned_graph(bpc)) && return 0
    bpc = copy(bpc)

    pes = vcat(path, [target_pe])
    incoming_es = boundary_partitionedges(bpc, pes)
    incoming_messages = ITensor[only(message(bpc, pe)) for pe in incoming_es]
    pvs = unique(vcat(src.(path), dst.(path)))
    vs = vertices(bpc, pvs)

    src_vs = vertices(bpc, src(target_pe))

    pe_linkinds = ITensorNetworks.linkinds(bpc, target_pe)
    pe_linkinds_sim = sim.(pe_linkinds)

    ts = ITensor[]

    for v in src_vs
        t = bpc[v]
        t_inds = filter(i -> i ∈ pe_linkinds, inds(t))
        if !isempty(t_inds)
            t_ind = only(t_inds)
            t_ind_pos = findfirst(x -> x == t_ind, pe_linkinds)
            t = replaceind(t, t_ind, pe_linkinds_sim[t_ind_pos])
        end
        push!(ts, t)
    end

    ts = ITensor[ts; ITensor[bpc[v] for v in setdiff(vs, src_vs)]]

    tensors = [ts; incoming_messages]
    seq = ITensorNetworks.contraction_sequence(tensors; alg = "einexpr", optimizer = Greedy())
    t = ITensors.contract(tensors; sequence = seq)

    row_combiner, col_combiner = ITensors.combiner(pe_linkinds), ITensors.combiner(pe_linkinds_sim)
    t = t * row_combiner * col_combiner
    t = array(t)
    eigvals, _ = eigen(t)
    eigvals = reverse(sort(eigvals; by = abs))

    length(eigvals) == 1 && return 0.0

    err = 1.0 - abs(eigvals[1]) / sum(abs.(eigvals))

    return err
end

function bp_error(bpc::BeliefPropagationCache, smallest_loop_size::Int; kwargs...)
    pg = partitioned_graph(bpc)
    cycles = NamedGraphs.cycle_to_path.(NamedGraphs.unique_simplecycles_limited_length(pg, smallest_loop_size))
    errs = []
    @show length(cycles)
    for path in cycles
        err_exact = bp_error_exact(bpc, PartitionEdge.(path[1:(length(path)-1)]), reverse(PartitionEdge(last(path))); kwargs...)
        push!(errs, err_exact)
    end
    return errs
end

ITensors.disable_warn_order()

function main()
    layers = [i for i in 1:20]
    topology = "Willow"

    no_loops = topology == "Willow" ? 78 : 25
    maxdim = 10

    errs = zeros((length(layers), no_loops))
    for (i, layer) in enumerate(layers)
        f = topology == "HeavyHex" ? "/mnt/home/jtindall/ceph/Data/HeisenbergDynamics/"*topology*"/wavefunction_muInf_Delta1.0_dt0.1_maxdim"*string(maxdim)*"_layer"*string(layer) : "/mnt/home/jtindall/ceph/Data/HeisenbergDynamics/"*topology*"/wavefunction_willow_muInf_Delta1.0_dt0.1_maxdim"*string(maxdim)*"_layer"*string(layer)
        ψ = last(deserialize(f))
        @show typeof(ψ)

        ψψ = build_bp_cache(ψ)
        loop_size=  topology == "HeavyHex" ? 12 : 4
        errs[i, :] = bp_error(ψψ, loop_size)
        println("Error was $(Statistics.mean(errs[i, :]))")
    end
    npzwrite("/mnt/home/jtindall/ceph/Data/HeisenbergDynamics/BPErrors/"*string(topology)*"Maxdim"*string(maxdim)*".npz", errs=  errs)

    @show layers, errs
end

main()