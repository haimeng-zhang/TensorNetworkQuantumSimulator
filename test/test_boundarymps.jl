@eval module $(gensym())
using ITensors: datatype
using NamedGraphs: edges
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using Random
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test BoundaryMPS" begin
    Random.seed!(123)
    g = named_grid((3, 3))

    #BMPS Cache
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetwork(eltype, g; bond_dimension = 2)
        ψ_BMPS = TN.BoundaryMPSCache(ψ, 4)
        @test TN.network(ψ_BMPS) isa TN.TensorNetwork
        @test ψ_BMPS isa TN.BoundaryMPSCache
        @test TN.graph(ψ_BMPS) == g
        @test datatype(ψ_BMPS) == datatype(ψ)
        @test TN.scalartype(ψ_BMPS) == TN.scalartype(ψ)

        ψ_BMPS = TN.update(ψ_BMPS)
        z_bmps = TN.partitionfunction(ψ_BMPS)
        @test z_bmps ≈ TN.contract(ψ; alg = "exact")
        @test z_bmps ≈ TN.contract(ψ; alg = "boundarymps", mps_bond_dimension = 4)
    end

    #BMPS Cache
    s = TN.siteinds("S=1", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetworkstate(eltype, g; bond_dimension = 2)
        ψ_BMPS = TN.BoundaryMPSCache(ψ, 4)
        @test TN.network(ψ_BMPS) isa TN.TensorNetworkState
        @test ψ_BMPS isa TN.BoundaryMPSCache
        @test TN.graph(ψ_BMPS) == g
        @test datatype(ψ_BMPS) == datatype(ψ)
        @test TN.scalartype(ψ_BMPS) == TN.scalartype(ψ)

        ψ_BMPS = TN.update(ψ_BMPS)
        z_bmps = TN.partitionfunction(ψ_BMPS)
        @test z_bmps ≈ TN.norm_sqr(ψ; alg = "exact")
        @test z_bmps ≈ TN.norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 4)
    end
end

end
