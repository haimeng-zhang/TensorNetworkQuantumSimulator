using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge
using TensorNetworkQuantumSimulator: dag, virtualinds
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta, random_itensor
using Dictionaries: Dictionary
using Random

function uniform_random_itensor(inds)
    t = ITensor(1.0, inds)
    for iv in eachindval(t)
        t[iv...] = rand()
    end
    return t
end

include("utils.jl")
include("update_rules.jl")

g = named_grid((10,10); periodic = false)
#Build physical site indices for spin-1/2 degrees of freedom
s = siteinds("S=1/2", g)

#Build a random TensorNetworkState on the graph with bond dimension 2
ψ = random_tensornetwork(Float64, g; bond_dimension = 2)
tensors = [uniform_random_itensor(inds(ψ[v])) for v in vertices(g)]
#T = TensorNetwork(Dictionary(collect(vertices(g)), tensors))
ψ = TensorNetworkState(Dictionary(collect(vertices(g)), tensors))
# #Take its dagger
ψdag = map_virtualinds(prime, map_tensors(dag, ψ))

# #Build the norm tensor network ψψ† and combine pairs of virtual inds
T = TensorNetwork(Dictionary(vertices(g), [ψ[v]*ψdag[v] for v in vertices(g)]))
TensorNetworkQuantumSimulator.combine_virtualinds!(T)

bs = construct_gbp_bs(T)
ms = construct_ms(bs)
ps = all_parents(ms, bs)
mobius_nos = mobius_numbers(ms, ps)
ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
cs = children(ms, ps, bs)
b_nos = calculate_b_nos(ms, ps, mobius_nos)

generalized_belief_propagation(T, bs, ms, ps, cs, b_nos; niters = 100, rate = 0.4)