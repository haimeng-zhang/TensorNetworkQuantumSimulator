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