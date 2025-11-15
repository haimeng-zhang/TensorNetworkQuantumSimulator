@eval module $(gensym())
using ITensors: datatype
using NamedGraphs: edges
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using Random
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test BP" begin
    Random.seed!(123)
    g = named_comb_tree((3, 3))

    #BP Cache
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetwork(eltype, g; bond_dimension = 2)
        ψ_BPC = TN.BeliefPropagationCache(ψ)
        @test TN.network(ψ_BPC) isa TN.TensorNetwork
        @test ψ_BPC isa TN.BeliefPropagationCache
        @test TN.graph(ψ_BPC) == g
        @test isempty(TN.messages(ψ_BPC))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test TN.scalartype(ψ_BPC) == TN.scalartype(ψ)

        ψ_BPC = TN.update(ψ_BPC)
        @test !isempty(TN.messages(ψ_BPC))
        @test length(keys(TN.messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = TN.partitionfunction(ψ_BPC)
        @test z_bp ≈ TN.contract(ψ; alg = "exact")
        @test z_bp ≈ TN.contract(ψ; alg = "bp")
    end

    #BP Cache
    s = TN.siteinds("S=1", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetworkstate(eltype, g; bond_dimension = 2)
        ψ_BPC = TN.BeliefPropagationCache(ψ)
        @test ψ_BPC isa TN.BeliefPropagationCache
        @test TN.network(ψ_BPC) isa TN.TensorNetworkState
        @test TN.graph(ψ_BPC) == g
        @test isempty(TN.messages(ψ_BPC))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test TN.scalartype(ψ_BPC) == TN.scalartype(ψ)

        ψ_BPC = TN.update(ψ_BPC)
        @test !isempty(TN.messages(ψ_BPC))
        @test length(keys(TN.messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = TN.partitionfunction(ψ_BPC)
        @test z_bp ≈ TN.norm_sqr(ψ; alg = "exact")
        @test z_bp ≈ TN.norm_sqr(ψ; alg = "bp")
    end
end

end
