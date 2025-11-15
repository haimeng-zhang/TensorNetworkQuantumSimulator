@eval module $(gensym())
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using Random
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test forms" begin
    Random.seed!(123)
    g = named_grid((3, 3))
    s = TN.siteinds("S=1/2", g)

    #Quadratic Form
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetworkstate(eltype, g, s; bond_dimension = 2)
        ψ = TN.normalize(ψ; alg = "bp")
        ψψ = TN.QuadraticForm(ψ)
        @test ψψ isa TN.QuadraticForm
        @test TN.scalartype(ψψ) == eltype
        @test TN.graph(ψψ) == g

        ψψ_bpc = TN.update(TN.BeliefPropagationCache(ψψ))
        @test TN.partitionfunction(ψψ_bpc) ≈ TN.norm_sqr(ψ; alg = "bp")

        mps_bond_dimension = 16
        ψψ_bmps = TN.update(TN.BoundaryMPSCache(ψψ, mps_bond_dimension))
        @test TN.partitionfunction(ψψ_bmps) ≈ TN.norm_sqr(ψ; alg = "exact")
    end

    #BiLinear Form
    g = named_comb_tree((3, 3))
    s = TN.siteinds("S=1/2", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetworkstate(eltype, g, s; bond_dimension = 3)
        ϕ = TN.random_tensornetworkstate(eltype, g, s; bond_dimension = 4)
        ψ, ϕ = TN.normalize(ψ; alg = "bp"), TN.normalize(ϕ; alg = "bp")
        ψϕ = TN.BilinearForm(ψ, ϕ)
        @test ψϕ isa TN.BilinearForm
        @test TN.scalartype(ψϕ) == eltype
        @test TN.graph(ψϕ) == g

        ψϕ_bpc = TN.update(TN.BeliefPropagationCache(ψϕ))

        @test TN.partitionfunction(ψϕ_bpc) ≈ TN.inner(ψ, ϕ; alg = "bp")
    end

end

end
