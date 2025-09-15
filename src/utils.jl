getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)

"""
    trace(Q::ITensorNetwork) 

Take the trace of an ITensorNetwork. In the Pauli basis this is the direct trace. In Schrodinger this is the sum of coefficients
"""
function trace(Q::ITensorNetwork)
    d = getphysicaldim(siteinds(Q))
    if d == 2
        vec = [1.0, 1.0]
    elseif d == 4
        vec = [1.0, 0.0, 0.0, 0.0]
    else
        throwdimensionerror()
    end

    val = ITensorNetworks.inner(ITensorNetwork(v -> vec, siteinds(Q)), Q; alg = "bp")
    return val
end

"""
    truncate(ψ::ITensorNetwork; maxdim, cutoff=nothing, bp_update_kwargs= (...))

Truncate the ITensorNetwork `ψ` to a maximum bond dimension `maxdim` using the specified singular value cutoff.
"""
function ITensorNetworks.truncate(
    ψ::ITensorNetwork;
    cache_update_kwargs = default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
    kwargs...,
)
    ψ_vidal = VidalITensorNetwork(ψ; kwargs...)
    return ITensorNetwork(ψ_vidal)
end

# Boundary MPS helpers for checking graph formats
function is_line_graph(g::AbstractGraph)
    vs = collect(vertices(g))
    nvs = length(vs)
    length(vs) == 1 && return true
    !is_tree(g) && return false
    ds = sort([degree(g, v) for v in vs])
    ds != vcat([1,1], [2 for d in 1:(nvs - 2)]) && return false
    return true
end

function is_ring_graph(g::AbstractGraph)
    g_mod = rem_edge(g, first(edges(g)))
    return is_line_graph(g_mod)
end