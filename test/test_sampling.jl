@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: datatype
using NamedGraphs: edges, vertices, vertextype
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_hexagonal_lattice_graph
using Random
using Statistics
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Sampling" begin
    Random.seed!(123)

    #Product State on hexagonal lattice
    g = named_hexagonal_lattice_graph(3, 3)
    ψ = TN.tensornetworkstate(ComplexF64, v -> "↑", g)
    ψ = TN.gauge_and_scale(ψ)
    bmps_sample = only(TN.sample(ψ, 1; alg = "boundarymps", norm_mps_bond_dimension = 1, projected_mps_bond_dimension = 1, gauge_and_scale = false))
    @test all([bmps_sample[v] == 0 for v in vertices(g)])

    bp_sample = only(TN.sample(ψ, 1; alg = "bp", gauge_state = false))
    @test all([bp_sample[v] == 0 for v in vertices(g)])

    #Random entangled state on square grid
    g = named_grid((3, 3))
    ψ = TN.random_tensornetworkstate(ComplexF64, g; bond_dimension = 2)
    #Set BP norm to 1 (sample will do this automatically unless you state gauge_and_scale = false)
    ψ = TN.gauge_and_scale(ψ)

    bp_samples = TN.sample(ψ, 10; alg = "bp", gauge_state = false)
    @test bp_samples isa Vector{<:Dictionary{vertextype(g), Int}}
    @test length(bp_samples) == 10
    @test all([keys(bp_sample) == vertices(g) for bp_sample in bp_samples])
    bmps_certified_samples = TN.sample_certified(ψ, 10; alg = "boundarymps", norm_mps_bond_dimension = 4, projected_mps_bond_dimension = 4, gauge_and_scale = false)
    p_qs = first.(bmps_certified_samples)

    #We importance sampled with big enough mps_dimensions. Standard deviation should be small, mean should be norm of network.
    @test Statistics.std(p_qs) < 1.0e-8
    @test Statistics.mean(p_qs) ≈ TN.norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 4)
end

end
