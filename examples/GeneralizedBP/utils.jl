using TensorNetworkQuantumSimulator
using NamedGraphs: unique_simplecycles_limited_length
using ITensors: Index

using Graphs: topological_sort
using Graphs.SimpleGraphs: SimpleDiGraph


function construct_bp_bs(t::AbstractTensorNetwork)
    return collect([[i for i in inds(t[v])] for v in vertices(t)])
end

function construct_gbp_bs(t::AbstractTensorNetwork)
    bs = construct_bp_bs(t)
    cycles = unique_simplecycles_limited_length(t, 4)
    gbp_bs = copy(bs)
    for cycle in cycles
        is = Index[]
        for (i, v) in enumerate(cycle)
            if i != length(cycle)
                index = only(virtualinds(t, NamedEdge(v => cycle[i+1])))
            else
                index = only(virtualinds(t, NamedEdge(v => cycle[1])))
            end
            push!(is, index)
        end
        push!(gbp_bs, is)  # Add first vertex to close the loop
    end

    return gbp_bs
end

function intersections(ms)
    intersects = []
    for i in 1:length(ms), j in i+1:length(ms)
        s = intersect(Set(ms[i]), Set(ms[j]))
        if !isempty(s) && s ∉ intersects
            push!(intersects, s)
        end
    end
    return collect.(intersects)
end

function construct_ms(bs)
    current_ms = intersections(bs)
    all_ms = []

    while !isempty(current_ms)
        for m in current_ms
            if Set(m) ∉ all_ms
                push!(all_ms, Set(m))
            end
        end
        current_ms = intersections(current_ms)
    end

    return collect.(all_ms)
end

function parents(m, bs)
    parents = []
    for (i, b) in enumerate(bs)
        if issubset(Set(m), Set(b))
            push!(parents, i)
        end
    end
    return parents
end

function all_parents(ms, bs)
    ms_parents = []
    for m in ms
        push!(ms_parents, parents(m, bs))
    end
    return ms_parents
end

#TODO: Figure this out
function mobius_numbers(ms, ps)
    #First get the subset matrix
    mat = zeros(Int, length(ms), length(ms))
    for (i, m1) in enumerate(ms), (j, m2) in enumerate(ms[(i + 1):end])
        if issubset(Set(m1), Set(m2))
            mat[i, j + i] = 1
        end
        if i != j && issubset(Set(m2), Set(m1))
            mat[j + i, i] = 1
        end
    end

    @show length(ms)
    @show ms[1], ms[2], ms[3]
    @show ps[3]
    g = SimpleDiGraph(mat)
    ts = topological_sort(g)
    ts = reverse(ts)
    
    mobius_numbers = zeros(Int, length(ms))
    for i in 1:length(ms)
        mobius_numbers[ts[i]] = 1 - length(ps[ts[i]])
        for l in 1:(i-1)
            if mat[ts[i], ts[l]] == 1
                mobius_numbers[ts[i]] = mobius_numbers[ts[i]] - mobius_numbers[ts[l]]
            end
        end
    end

    @show mobius_numbers
    return mobius_numbers
end

# function initialize_messages(ms)
#     messages = []
#     for m in ms
#         msg = ITensor(1.0, m)
#         push!(messages, msg)
#     end
#     return messages
# end