using CUDA
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using ITensorNetworks
const ITN = ITensorNetworks
using ITensors
using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using EinExprs: Greedy
using LinearAlgebra
using Serialization
using Random
Random.seed!(1634)
function main()
    g = named_grid((2, 2))
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(ComplexF32, v -> "↑", s)
    ψ = CUDA.cu(ψ)
    nsamples = 10
    projected_message_rank = maxlinkdim(ψ)
    norm_message_rank = 10
    probs_and_bitstrings = TN.sample_directly_certified(ψ, nsamples; norm_message_rank = norm_message_rank)
    println("Successfully generated $(length(probs_and_bitstrings)) samples")
    return probs_and_bitstrings
end
main()