using NamedGraphs.PartitionedGraphs: PartitionedGraph, partitions_graph, partitionvertices, PartitionEdge, partitionedges, partitionedge, PartitionVertex
using NamedGraphs: add_edges!

struct BoundaryMPSCache{V, BPC<:BeliefPropagationCache{V}} <: AbstractBeliefPropagationCache{V}
    bp_cache::BPC
    supergraph::PartitionedGraph
    sorted_edges::Dictionary
    mps_bond_dimension::Int64
end

default_update_alg(bmps_cache::BoundaryMPSCache) = "bp"
function set_default_kwargs(alg::Algorithm"bp", bmps_cache::BoundaryMPSCache)
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bmps_cache))
    edge_sequence = get(alg.kwargs, :edge_sequence, default_bp_edge_sequence(bmps_cache))
    message_update_alg = set_default_kwargs(
        get(alg.kwargs, :message_update_alg, Algorithm(default_message_update_alg(bmps_cache))), bmps_cache
    )
    return Algorithm("bp"; maxiter, edge_sequence, message_update_alg, tolerance = nothing)
end

function default_bp_edge_sequence(bmps_cache::BoundaryMPSCache)
    return PartitionEdge.(forest_cover_edge_sequence(partitions_graph(supergraph(bmps_cache))))
end
default_bp_maxiter(bmps_cache::BoundaryMPSCache) = is_tree(partitions_graph(supergraph(bmps_cache))) ? 1 : 5
function default_message_update_alg(bmps_cache::BoundaryMPSCache)
    network(bp_cache(bmps_cache)) isa TensorNetworkState && return "orthogonal"
    network(bp_cache(bmps_cache)) isa ITensorNetwork && return "ITensorMPS"
end

default_normalize(alg::Algorithm"orthogonal") = true
default_tolerance(bmps_cache::BoundaryMPSCache) = default_tolerance(ITensors.NDTensors.scalartype(bmps_cache))
_default_boundarymps_update_niters = 50
function set_default_kwargs(alg::Algorithm"orthogonal", bmps_cache::BoundaryMPSCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    tolerance = get(alg.kwargs, :tolerance, default_tolerance(bmps_cache))
    niters = get(alg.kwargs, :niters,  _default_boundarymps_update_niters)
    return Algorithm("orthogonal"; tolerance, niters, normalize)
end

default_normalize(alg::Algorithm"ITensorMPS") = true
function set_default_kwargs(alg::Algorithm"ITensorMPS", bmps_cache::BoundaryMPSCache)
    cutoff = get(alg.kwargs, :cutoff, _default_boundarymps_update_cutoff)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    return Algorithm("ITensorMPS"; cutoff, normalize)
end

# const _default_boundarymps_update_alg = "orthogonal"
# const _default_boundarymps_update_niters = 40
# const _default_boundarymps_update_tolerance = 1e-12
# const _default_boundarymps_update_cutoff = 1e-12

# function default_boundarymps_update_kwargs(; cache_is_flat = false, kwargs...)
#     message_update_alg = Algorithm(ITensorNetworks.default_message_update_alg(cache_is_flat))
#     return (; message_update_alg, default_message_update_kwargs(; cache_is_flat, kwargs...)...)
# end

# ITensorNetworks.default_message_update_alg(cache_is_flat::Bool = false) = cache_is_flat ? "ITensorMPS" : "orthogonal"

# function default_message_update_kwargs(; cache_is_flat = false, cutoff = _default_boundarymps_update_cutoff, kwargs...)
#     !cache_is_flat && return return (; niters = _default_boundarymps_update_niters, tolerance = _default_boundarymps_update_tolerance)
#     return (; cutoff = cutoff, kwargs...)
# end

function is_correct_format(bmps_cache::BoundaryMPSCache)
    s = supergraph(bmps_cache)
    effective_graph = partitions_graph(s)
    if !is_ring_graph(effective_graph) && !is_line_graph(effective_graph)
        error("Upon partitioning, graph does not form a line or ring: can't run boundary MPS")
    end
    for pv in partitionvertices(s)
        if !is_line_graph(subgraph(s, pv))
            error("There's a partition that does not form a line: can't run boundary MPS")
        end   
    end
    return true
end

# default_cache_update_kwargs(alg::Algorithm"boundarymps") = default_boundarymps_update_kwargs()

# ITensorNetworks.default_update_alg(bmps_cache::BoundaryMPSCache) = "bp"
# function ITensorNetworks.set_default_kwargs(alg::Algorithm"bp", bmps_cache::BoundaryMPSCache)
#     maxiter = get(alg.kwargs, :maxiter, is_tree(partitions_graph(ppg(bmps_cache))) ? 1 : nothing)
#     edge_sequence = get(alg.kwargs, :edge_sequence, pair.(default_edge_sequence(ppg(bmps_cache))))
#     verbose = get(alg.kwargs, :verbose, false)
#     tol = get(alg.kwargs, :tol, nothing)
#     message_update_alg = ITensorNetworks.set_default_kwargs(get(alg.kwargs, :message_update_alg, Algorithm(ITensorNetworks.default_message_update_alg(is_flat(bmps_cache)))))
#     return Algorithm("bp"; tol, message_update_alg, maxiter, edge_sequence, verbose)
# end

bp_cache(bmps_cache::BoundaryMPSCache) = bmps_cache.bp_cache
supergraph(bmps_cache::BoundaryMPSCache) = bmps_cache.supergraph
mps_bond_dimension(bmps_cache::BoundaryMPSCache) = bmps_cache.mps_bond_dimension
sorted_edges(bmps_cache::BoundaryMPSCache) = bmps_cache.sorted_edges
function sorted_edges(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    return sorted_edges(bmps_cache)[pe]
end

#Forward onto the BP Cache
for f in [
    :(messages),
    :(ITensors.datatype),
    :(ITensorNetworks.linkinds),
    :(ITensors.NDTensors.scalartype),
    :(network)
]
@eval begin
    function $f(bmps_cache::BoundaryMPSCache, args...; kwargs...)
        return $f(bp_cache(bmps_cache), args...; kwargs...)
    end
end
end

default_message(bmps_cache::BoundaryMPSCache, e::NamedEdge) = default_message(bp_cache(bmps_cache), e)
setmessage!(bmps_cache::BoundaryMPSCache, e::NamedEdge, m::ITensor) = setmessage!(bp_cache(bmps_cache), e, m)
deletemessages!(bmps_cache::BoundaryMPSCache, es::Vector{<:NamedEdge}) = deletemessages!(bp_cache(bmps_cache), es)
message(bmps_cache::BoundaryMPSCache, e::NamedEdge) = message(bp_cache(bmps_cache), e)

# function ITensorNetworks.default_bp_maxiter(
#     alg::Algorithm,
#     bmps_cache::BoundaryMPSCache,
# )
#     return default_bp_maxiter(partitions_graph(ppg(bmps_cache)))
# end
# function ITensorNetworks.default_edge_sequence(alg::Algorithm, bmps_cache::BoundaryMPSCache)
#     return pair.(default_edge_sequence(ppg(bmps_cache)))
# end

# default_boundarymps_message_rank(tn::AbstractITensorNetwork) = maxlinkdim(tn)^2
# ITensorNetworks.partitions(bmps_cache::BoundaryMPSCache) =
#     parent.(collect(partitionvertices(ppg(bmps_cache))))
# NamedGraphs.PartitionedGraphs.partitionedges(bmps_cache::BoundaryMPSCache) = pair.(partitionedges(ppg(bmps_cache)))

# function ITensorNetworks.cache(
#     alg::Algorithm"boundarymps",
#     tn;
#     bp_cache_construction_kwargs = default_cache_construction_kwargs(Algorithm("bp"), tn),
#     kwargs...,
# )
#     return BoundaryMPSCache(
#         BeliefPropagationCache(tn; bp_cache_construction_kwargs...);
#         kwargs...,
#     )
# end

# function ITensorNetworks.default_cache_construction_kwargs(alg::Algorithm"boundarymps", tn)
#     return (;
#         bp_cache_construction_kwargs = default_cache_construction_kwargs(
#             Algorithm("bp"),
#             tn,
#         )
#     )
# end

function Base.copy(bmps_cache::BoundaryMPSCache)
    return BoundaryMPSCache(
        copy(bp_cache(bmps_cache)),
        copy(supergraph(bmps_cache)),
        copy(sorted_edges(bmps_cache)),
        mps_bond_dimension(bmps_cache),
    )
end

#Get the dimension of the virtual index between the two message tensors on pe1 and pe2
function virtual_index_dimension(
    bmps_cache::BoundaryMPSCache,
    e1::NamedEdge,
    e2::NamedEdge,
)
    s = supergraph(bmps_cache)
    es = sorted_edges(bmps_cache, partitionedge(s, e1))

    if findfirst(x -> x == e1, es) > findfirst(x -> x == e2, es)
        lower_e, upper_e = e2, e1
    else
        lower_e, upper_e = e1, e2
    end

    inds_above = reduce(vcat, linkinds.((bmps_cache,), edges_above(bmps_cache, lower_e)))
    inds_below = reduce(vcat, linkinds.((bmps_cache,), edges_below(bmps_cache, upper_e)))

    x1 = prod(Float64.(dim.(inds_above)))
    x2 = prod(Float64.(dim.(inds_below)))
    if network(bmps_cache) isa TensorNetworkState
        return Int(minimum((x1*x1, x2*x2,Float64(mps_bond_dimension(bmps_cache)))))
    else
        return Int(minimum((x1, x2,Float64(mps_bond_dimension(bmps_cache)))))
    end
end

# #Vertices of the planargraph
# function planargraph_vertices(bmps_cache::BoundaryMPSCache, partition)
#     return vertices(ppg(bmps_cache), PartitionVertex(partition))
# end

# #Get partition(s) of vertices of the planargraph
# function planargraph_partitions(bmps_cache::BoundaryMPSCache, vertices::Vector)
#     return parent.(partitionvertices(ppg(bmps_cache), vertices))
# end

# function planargraph_partition(bmps_cache::BoundaryMPSCache, vertex)
#     return only(planargraph_partitions(bmps_cache, [vertex]))
# end

#Constructor, inserts missing edge in the planar graph to ensure each partition is connected
#allowing the code to work for arbitrary grids and not just square grids
function BoundaryMPSCache(
    bpc::BeliefPropagationCache;
    grouping_function::Function = v -> first(v),
    group_sorting_function::Function = v -> last(v),
    mps_bond_dimension::Int,

)
    pseudo_edges = pseudo_planar_edges(bpc; grouping_function)
    planar_graph = underlying_graph(bpc)
    NamedGraphs.add_edges!(planar_graph, pseudo_edges)
    vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
    vertex_groups = map(x -> sort(x; by = group_sorting_function), vertex_groups)
    supergraph = PartitionedGraph(planar_graph, vertex_groups)
    pes = vcat(partitionedges(supergraph), reverse.(partitionedges(supergraph)))
    sorted_es = Dictionary(pes, [sorted_edges(supergraph, pe) for pe in pes])

    bmps_cache = BoundaryMPSCache(bpc, supergraph, sorted_es, mps_bond_dimension)
    @assert is_correct_format(bmps_cache)
    set_interpartition_messages!(bmps_cache, pes)

    return bmps_cache
end

BoundaryMPSCache(tns::Union{ITensorNetwork, TensorNetworkState}, args...; kwargs...) = BoundaryMPSCache(BeliefPropagationCache(tns); kwargs...)

all_partitionedges(bmps_cache::BoundaryMPSCache) = vcat(partitionedges(supergraph(bmps_cache)), reverse.(partitionedges(supergraph(bmps_cache))))

#Initialise all the interpartition message tensors
function set_interpartition_messages!(
    bmps_cache::BoundaryMPSCache,
    partitionedges::Vector{<:PartitionEdge} = all_partitionedges(bmps_cache),
)
    m_keys = keys(messages( bmps_cache))
    for pe in partitionedges
        es = sorted_edges(bmps_cache, pe)
        for e in es
            if e ∉ m_keys
                setmessage!(bmps_cache, e, default_message(bmps_cache, e))
            end
        end
        for i = 1:(length(es)-1)
            virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i+1])
            ind = Index(virt_dim, "m$(i)$(i+1)")
            m1, m2 = message(bmps_cache, es[i]), message(bmps_cache, es[i+1])
            t = adapt(datatype(m1))(dense(delta(ind)))
            setmessage!(bmps_cache, es[i], m1*t)
            setmessage!(bmps_cache, es[i+1], m2*t)
        end
    end
    return bmps_cache
end

#Switch the message tensors on partition edges with their reverse (and dagger them)
function switch_message!(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    ms = messages(bmps_cache)
    me, mer = message(bmps_cache, e), message(bmps_cache, reverse(e))
    set!(ms, e, dag(mer))
    set!(ms, reverse(e), dag(me))
    return bmps_cache
end

function switch_messages!(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    for pe in sorted_edges(bmps_cache, pe)
        switch_message!(bmps_cache, pe)
    end
    return bmps_cache
end

function partition_graph(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
    vs = vertices(supergraph(bmps_cache), partition)
    es = filter(e -> src(e) ∈ vs && dst(e) ∈  vs, edges(supergraph(bmps_cache)))
    g = NamedGraph(vs)
    add_edges!(g, es)
    return g
end

function partition_update!(bmps_cache::BoundaryMPSCache, seq::Vector{<:NamedEdge})
    alg = set_default_kwargs(Algorithm("contract", normalize = false, enforce_hermiticity = false), bp_cache(bmps_cache))
    for e in seq
        m = updated_message(alg, bp_cache(bmps_cache), e)
        setmessage!(bmps_cache, e, m)
    end
    return bmps_cache
end

function partition_update(bmps_cache::BoundaryMPSCache, seq::Vector{<:NamedEdge})
    bmps_cache = copy(bmps_cache)
    return partition_update!(bmps_cache, seq)
end

# #Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2
function gauge_step!(
    alg::Algorithm"orthogonal",
    bmps_cache::BoundaryMPSCache,
    e1::NamedEdge,
    e2::NamedEdge;
    kwargs...,
)
    m1, m2 = message(bmps_cache, e1), message(bmps_cache, e2)
    @assert !isempty(commoninds(m1, m2))
    left_inds = uniqueinds(m1, m2)
    m1, Y = factorize(m1, left_inds; ortho = "left", kwargs...)
    m2 = m2 * Y
    setmessage!(bmps_cache, e1, m1)
    setmessage!(bmps_cache, e2, m2)
    return bmps_cache
end

#Move the orthogonality centre via a sequence of steps between message tensors
function gauge_walk!(
    alg::Algorithm,
    bmps_cache::BoundaryMPSCache,
    seq::Vector;
    kwargs...,
)
    for (e1, e2) in seq
        gauge_step!(alg::Algorithm, bmps_cache, e1, e2; kwargs...)
    end
    return bmps_cache
end

function inserter!(
    alg::Algorithm,
    bmps_cache::BoundaryMPSCache,
    update_e::NamedEdge,
    m::ITensor;
)
    setmessage!(bmps_cache, reverse(update_e), dag(m))
    return bmps_cache
end

#Default 1-site extracter
function extracter(
    alg::Algorithm"orthogonal",
    bmps_cache::BoundaryMPSCache,
    update_e::NamedEdge;
)
    message_update_alg = set_default_kwargs(Algorithm("contract"; normalize = false, enforce_hermiticity = false), bp_cache(bmps_cache))
    m = updated_message(message_update_alg, bp_cache(bmps_cache),update_e)
    return m
end

# function ITensors.commonind(bmps_cache::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge)
#     m1, m2 = message(bmps_cache, pe1), message(bmps_cache, pe2)
#     return commonind(only(m1), only(m2))
# end

# function merge_internal_tensors(O::Union{MPS, MPO})
#     internal_inds = filter(i -> isempty(ITensorMPS.siteinds(O, i)), [i for i in 1:length(O)])

#     while !isempty(internal_inds)
#         site = first(internal_inds)
#         tensors = [O[i] for i in setdiff([i for i in 1:length(O)], [site])]
#         if site != length(O)
#             tensors[site] = tensors[site] * O[site]
#         else
#             tensors[site - 1] = tensors[site - 1] * O[site]
#         end

#         O = typeof(O)(tensors)

#         internal_inds = filter(i -> isempty(ITensorMPS.siteinds(O, i)), [i for i in 1:length(O)])
#     end
#     return O
# end

# function ITensorMPS.MPO(bmps_cache::BoundaryMPSCache, partition)
#     sorted_vs = sort(planargraph_vertices(bmps_cache, partition))
#     ts = [copy(bmps_cache[v]) for v in sorted_vs]
#     O = ITensorMPS.MPO(ts)
#     return O
# end

# function ITensorMPS.MPS(bmps_cache::BoundaryMPSCache, partitionpair::Pair)
#     sorted_pes = planargraph_sorted_partitionedges(bmps_cache, partitionpair)
#     ms = [only(message(bmps_cache, pe)) for pe in sorted_pes]
#     return ITensorMPS.MPS(ms)
# end

# function truncate!(bmps_cache::BoundaryMPSCache, partitionpair::Pair; truncate_kwargs...)
#     M = ITensorMPS.MPS(bmps_cache, partitionpair)
#     M = ITensorMPS.truncate(M; truncate_kwargs...)
#     return set_interpartition_message!(bmps_cache, M, partitionpair)
# end

# function set_interpartition_message!(bmps_cache::BoundaryMPSCache, M::Union{MPS, MPO}, partitionpair::Pair)
#     sorted_pes = planargraph_sorted_partitionedges(bmps_cache, partitionpair)
#     for i in 1:length(M)
#         set_message!(bmps_cache, sorted_pes[i], ITensor[M[i]])
#     end
#     return bmps_cache
# end

function updater!(alg::Algorithm"orthogonal", bmps_cache::BoundaryMPSCache, partition_graph::AbstractGraph, prev_e, update_e)
    prev_e == nothing && return bmps_cache

    gauge_step!(alg, bmps_cache, reverse(prev_e), reverse(update_e))
    update_seq = a_star(partition_graph, src(prev_e), src(update_e))
    partition_update!(bmps_cache, update_seq)
    return bmps_cache
end
  
function update_message!(
    alg::Algorithm"orthogonal", bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
  delete_partition_messages!(bmps_cache, src(pe))
  switch_messages!(bmps_cache, pe)
  es = sorted_edges(bmps_cache, pe)
  g = partition_graph(bmps_cache, src(pe))
  update_seq = vcat([es[i] for i in 1:length(es)], [es[i] for i in (length(es) - 1):-1:2])

  init_gauge_seq = [(reverse(es[i]), reverse(es[i-1])) for i in length(es):-1:2]
  init_update_seq = post_order_dfs_edges(g, src(first(update_seq)))
  !isempty(init_gauge_seq) && gauge_walk!(alg, bmps_cache, init_gauge_seq)
  !isempty(init_update_seq) && partition_update!(bmps_cache, init_update_seq)

  prev_cf, prev_e = 0, nothing
  for i = 1:alg.kwargs.niters
      cf = 0
      if i == alg.kwargs.niters
         update_seq = vcat(update_seq, es[1])
      end
      for update_e in update_seq
          updater!(alg, bmps_cache, g, prev_e, update_e)
          m = extracter(alg, bmps_cache, update_e)
          n = norm(m)
          cf += n
          if alg.kwargs.normalize
              m /= n
          end
          inserter!(alg, bmps_cache, update_e, m)
          prev_e = update_e
      end
      cf /= length(update_seq)
      epsilon = abs(cf - prev_cf)
      !isnothing(alg.kwargs.tolerance) && epsilon < alg.kwargs.tolerance && break
      prev_cf = cf
  end
  delete_partition_messages!(bmps_cache, src(pe))
  switch_messages!(bmps_cache, pe)
  return bmps_cache
end

# function prev_partitionpair(bmps_cache::BoundaryMPSCache, partitionpair::Pair)
#     pppg = partitions_graph(ppg(bmps_cache))
#     vns = neighbors(pppg, first(partitionpair))
#     length(vns) == 1 && return nothing

#     @assert length(vns) == 2
#     v1, v2 = first(vns), last(vns)
#     last(partitionpair) == v1 && return v2 => first(partitionpair)
#     last(partitionpair) == v2 && return v1 => first(partitionpair)
# end

# function generic_apply(O::MPO, M::MPS; normalize = true, kwargs...)
#     is_simple_mpo = (length(O) == length(M) && all([length(ITensors.siteinds(O, i)) == 2 for i in 1:length(O)]))

#     if is_simple_mpo
#         out = ITensorMPS.apply(O, M; alg = "naive", kwargs...)
#         if normalize
#             out = ITensors.normalize(out)
#         end
#         return out
#     end

#     O_tensors = ITensor[]
#     for i in 1:length(O)
#         m_ind = filter(j -> !isempty(ITensors.commoninds(O[i], M[j])), [j for j in 1:length(M)])
#         if isempty(m_ind)
#             push!(O_tensors, O[i])
#         else
#             m_ind = only(m_ind)
#             push!(O_tensors, O[i] * M[m_ind])
#         end
#     end
#     O = ITensorNetwork([i for i in 1:length(O_tensors)], O_tensors)
    
#     #Transform away edges that make a loop
#     loop_edges = filter(e -> abs(src(e) - dst(e)) != 1, edges(O))
#     for e in loop_edges
#         edge_to_split = e
#         inbetween_vertices = [i for i in (minimum((src(e), dst(e)))+1):(maximum((src(e), dst(e)))-1)]
#         for v in inbetween_vertices
#             edge_to_split_ind = only(linkinds(O, edge_to_split))
#             O = ITensorNetworks.split_index(O, [edge_to_split])
#             d = adapt(datatype(O[v]))(denseblocks(delta(edge_to_split_ind, edge_to_split_ind')))
#             O[v] *= d
#             edge_to_split = NamedEdge(v => maximum((src(e), dst(e))))
#         end
#     end

#     O = ITensorNetworks.combine_linkinds(O)
#     @assert is_tree(O)
#     O = ITensorMPS.MPS([O[v] for v in vertices(O)])
#     O = merge_internal_tensors(O)
    
#     if normalize
#         O = ITensors.normalize(O)
#     end

#     return truncate(O; kwargs...)
# end

# #Update all the message tensors on an interpartition via the ITensorMPS apply function
# function ITensorNetworks.update_message(
#     alg::Algorithm"ITensorMPS",
#     bmps_cache::BoundaryMPSCache,
#     partitionpair::Pair;
#     maxdim::Int = maximum_virtual_dimension(bmps_cache),
# )
#     bmps_cache = copy(bmps_cache)
#     prev_pp = prev_partitionpair(bmps_cache, partitionpair)
#     O = ITensorMPS.MPO(bmps_cache, first(partitionpair))
#     O = ITensorMPS.truncate(O; alg.kwargs.cutoff, maxdim)
#     if isnothing(prev_pp)
#         O = merge_internal_tensors(O)
#         if alg.kwargs.normalize 
#             O = ITensors.normalize(O)
#         end
#         return set_interpartition_message!(bmps_cache, O, partitionpair)
#     end

#     M = ITensorMPS.MPS(bmps_cache, prev_pp)
#     M_out = generic_apply(O, M; cutoff = alg.kwargs.cutoff, normalize =  alg.kwargs.normalize, maxdim)
#     return set_interpartition_message!(bmps_cache, M_out, partitionpair)
# end

# #Environment support, assume all vertices live in the same partition for now
# function ITensorNetworks.environment(bmps_cache::BoundaryMPSCache, verts::Vector; kwargs...)
#     vs = parent.((partitionvertices(bp_cache(bmps_cache), verts)))
#     partition = only(planargraph_partitions(bmps_cache, parent.(partitionvertices(bmps_cache, verts))))
#     pg = partition_graph(bmps_cache, partition)
#     update_seq = post_order_dfs_edges(pg,first(vs))
#     bmps_cache = partition_update(bmps_cache, PartitionEdge.(update_seq))
#     return environment(bp_cache(bmps_cache), verts; kwargs...)
# end

function vertex_scalar(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
    g = partition_graph(bmps_cache, partition)
    v = first(center(g))
    update_seq = post_order_dfs_edges(g,v)
    bmps_cache = partition_update(bmps_cache, update_seq)
    return vertex_scalar(bp_cache(bmps_cache), v)
end

function edge_scalar(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    es = sorted_edges(bmps_cache, pe)
    out = ITensor(one(Bool))
    for e in es
        out = (out * (message(bmps_cache, e))) * message(bmps_cache, reverse(e))
    end
    return out[]
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
    g = partition_graph(bmps_cache, partition)
    es = edges(g)
    es = vcat(es, reverse.(es))
    return deletemessages!(bmps_cache, filter(e -> e ∈ keys(messages(bmps_cache)), es))
end


function vertex_scalars(
    bmps_cache::BoundaryMPSCache, vertices = partitionvertices(supergraph(bmps_cache)); kwargs...
)
    return map(v -> vertex_scalar(bmps_cache, v; kwargs...), vertices)
end

function edge_scalars(
    bmps_cache::BoundaryMPSCache, edges = partitionedges(supergraph(bmps_cache)); kwargs...
)
return map(e -> edge_scalar(bmps_cache, e; kwargs...), edges)
end

# function delete_partitionpair_messages!(bmps_cache::BoundaryMPSCache, partitionpair::Pair)
#     pes = planargraph_sorted_partitionedges(bmps_cache, partitionpair)
#     return delete_messages!(bmps_cache, filter(pe -> pe ∈ keys(messages(bmps_cache)), pes))
# end


#PartitionedGraph Helpers
#Add edges necessary to connect up all vertices in a partition in the planar graph created by the sort function
function pseudo_planar_edges(
    bpc::BeliefPropagationCache;
    grouping_function = v -> first(v),
)
    g = underlying_graph(bpc)
    partitions = unique(grouping_function.(collect(vertices(g))))
    pseudo_edges = NamedEdge[]
    for p in partitions
        vs = sort(filter(v -> grouping_function(v) == p, collect(vertices(g))))
        for i = 1:(length(vs)-1)
            if vs[i] ∉ neighbors(g, vs[i+1])
                push!(pseudo_edges, NamedEdge(vs[i] => vs[i+1]))
            end
        end
    end
    return pseudo_edges
end

#Functions to get the parellel edges sitting above and below a edge
function edges_above(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es = sorted_edges(bmps_cache, partitionedge(supergraph(bmps_cache), e))
    e_pos = only(findall(x -> x == e, es))
    return NamedEdge[es[i] for i = (e_pos+1):length(es)]
end

function edges_below(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es = sorted_edges(bmps_cache, partitionedge(supergraph(bmps_cache), e))
    e_pos = only(findall(x -> x == e, es))
    return NamedEdge[es[i] for i = 1:(e_pos-1)]
end

function edge_above(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es_above = edges_above(bmps_cache, e)
    isempty(es_above) && return nothing
    return first(es_above)
end

function edge_below(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es_below = edges_below(bmps_cache, e)
    isempty(es_below) && return nothing
    return last(es_below)
end

#Sort (bottom to top) edges between pair of partitions in the planargraph
function sorted_edges(pg::PartitionedGraph, pe::PartitionEdge)
    src_vs, dst_vs = vertices(pg, src(pe)),vertices(pg, dst(pe))
    es = reduce(
        vcat,
        [
            [src_v => dst_v for dst_v in intersect(neighbors(pg, src_v), dst_vs)] for
            src_v in src_vs
        ],
    )
    return sort(NamedEdge.(es); by = x -> findfirst(isequal(src(x)), src_vs))
end