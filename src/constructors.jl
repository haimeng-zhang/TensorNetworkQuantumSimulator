const stringtostatemap = Dict("I" => [1, 0, 0, 0], "X" => [0,1,0,0], "Y" => [0,0,1,0], "Z" => [0,0,0,1])

"""
    zerostate(g::NamedGraph)

Tensor network for vacuum state on given graph, i.e all spins up
"""
function zerostate(eltype, g::NamedGraph)
    return tensornetworkstate(eltype, v->"↑", g, "S=1/2")
end

zerostate(g::NamedGraph) = zerostate(Float64, g)

"""
    topaulitensornetwork(op, g::NamedGraph)

Tensor network (in Heisenberg picture). Function should map vertices of the graph to pauli strings.
"""
function paulitensornetworkstate(eltype, f::Function, g::NamedGraph)
    h = v -> stringtostatemap[f(v)]
    return tensornetworkstate(eltype, h, g, "Pauli")
end

topaulitensornetwork(f::Function, g::NamedGraph) = topaulitensornetwork(Float64, f, g)

"""
    identitytensornetwork(tninds::IndsNetwork)

Tensor network (in Heisenberg picture) for identity matrix on given IndsNetwork
"""
function identitytensornetworkstate(eltype, g::NamedGraph)
    return paulitensornetworkstate(eltype, v -> "I", g)
end

identitytensornetworkstate(g::NamedGraph) = identitytensornetworkstate(Float64, g)