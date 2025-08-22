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
    χ = 2
    message_rank = 2
    g = named_grid((nx, ny))
    s = siteinds("S=1/2", g)
    println("Building random $(nx) x $(ny) tensor network of bond dimension $(χ)")
    ψ = ITensorNetworks.random_tensornetwork(ComplexF32, s; link_space = χ)

    ψ = ITensorNetworks.normalize(ψ; alg = "bp", cache_update_kwargs = (; maxiter = 10))

    # t1 = time()
    # ψIψ = build_normsqr_bp_cache(ψ, message_rank; cache_update_kwargs = (; message_update_alg = Algorithm("orthogonal"; niters = 40, tolerance = nothing)))
    # z = ITensorNetworks.scalar(ψIψ)
    # t2 = time()
    # println("Boundary MPS using MPS bond dimension of $(message_rank) took $(t2- t1) secs on CPU")
    # println("CPU computed value for the norm is $(z)")

    t1 = time()
    ψIψ = build_normsqr_bp_cache(Metal.mtl(ψ), message_rank;  cache_update_kwargs = (; message_update_alg = Algorithm("orthogonal"; niters = 40, tolerance = nothing)))
    z = ITensorNetworks.scalar(ψIψ)
    t2 = time()
    println("Boundary MPS using MPS bond dimension of $(message_rank) took $(t2- t1) secs on GPU")
    println("GPU computed value for the norm is $(z)")

end

main()
