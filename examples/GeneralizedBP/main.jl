using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge
using TensorNetworkQuantumSimulator: dag, virtualinds
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta
using Dictionaries: Dictionary

include("utils.jl")

g = named_grid((3, 3); periodic = false)
#Build physical site indices for spin-1/2 degrees of freedom
s = siteinds("S=1/2", g)

#Build a random TensorNetworkState on the graph with bond dimension 2
ψ = random_tensornetworkstate(ComplexF32, g, s; bond_dimension = 2)
# #Take its dagger
ψdag = map_virtualinds(prime, map_tensors(dag, ψ))

# #Build the norm tensor network ψψ† and combine pairs of virtual inds
T = TensorNetwork(Dictionary(vertices(g), [ψ[v]*ψdag[v] for v in vertices(g)]))
TensorNetworkQuantumSimulator.combine_virtualinds!(T)

#bs = construct_gbp_bs(T)
bs = construct_gbp_bs(T)

@show length(bs)
ms = construct_ms(bs)

ps = all_parents(ms, bs)

mobius_nos = mobius_numbers(ms, ps)

@show mobius_nos