using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors: datatype, Algorithm

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using EinExprs: Greedy

using Metal

using Random
Random.seed!(1634)

function main()
    nx, ny = 3,3
    χ = 3
    g = named_grid((nx, ny))
    s = siteinds("S=1/2", g)
    ψ = ITensorNetworks.random_tensornetwork(s; link_space = χ)

    t1 = time()
    ψIψ = build_bp_cache(ψ)
    t2 = time()
    println("Took $(t2- t1) secs on CPU")
    @show scalar(ψIψ)

    t1 = time()
    ψIψ = build_bp_cache(Metal.mtl(ψ))
    t2 = time()
    println("Took $(t2- t1) secs on GPU")
    @show scalar(ψIψ)

end

main()
