@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: ITensors, Index, inds
using NamedGraphs: NamedGraph, vertices, degree, rem_vertex!
using NamedGraphs.GraphsExtensions: rem_vertex, add_edge
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_hexagonal_lattice_graph, named_path_graph
using Random
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Constructors" begin
    Random.seed!(123)

    #TensorNetwork construction from tensors
    i,j,k,l = Index(2), Index(2), Index(2), Index(2)
    A,B,C,D = ITensors.random_itensor(i,j), ITensors.random_itensor(j,k), ITensors.random_itensor(k,l), ITensors.random_itensor(l,i)
    t = TN.TensorNetwork([A,B,C,D])
    @test t isa TN.TensorNetwork
    @test TN.scalartype(t) == eltype(A)
    @test TN.maxvirtualdim(t) == 2
    @test TN.graph(t) isa NamedGraph
    @test TN.graph(t) == add_edge(named_path_graph(4), 1=>4)

    #TensorNetwork pre-defined constructor
    g = named_hexagonal_lattice_graph(3,3)
    χ = 3
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetwork(eltype, g; bond_dimension = χ)
        @test ψ isa TN.TensorNetwork
        @test TN.scalartype(ψ) == eltype
        @test TN.graph(ψ) == g
        @test TN.maxvirtualdim(ψ) == 3
        @test all([length(inds(ψ[v])) == degree(g, v) for v in vertices(ψ)])

        v = first(vertices(g))
        rem_vertex!(ψ, v)
        @test TN.graph(ψ) == rem_vertex(g, v)
        @test !all([length(inds(ψ[v])) == degree(g, v) for v in vertices(ψ)])
    end

    #SiteInds
    s = TN.siteinds("S=1/2", g)
    @test s isa Dictionary
    @test keys(s) == vertices(g)
    @test all([s[v] isa Vector{<:Index} for v in vertices(g)])
    @test all([length(s[v]) == 1 for v in vertices(g)])

    #TensorNetworkState
    χ = 3
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = TN.random_tensornetworkstate(eltype, g, s; bond_dimension = χ)
        @test ψ isa TN.TensorNetworkState
        @test TN.scalartype(ψ) == eltype
        @test TN.siteinds(ψ) == s
        @test TN.graph(ψ) == g
        @test TN.maxvirtualdim(ψ) == 3
        @test all([length(inds(ψ[v])) == degree(g, v) + 1 for v in vertices(ψ)])
        @test all([TN.siteinds(ψ, v) == s[v] for v in vertices(ψ)])

        ψ = TN.tensornetworkstate(eltype, v -> "X+", g, "S=1/2")
        @test TN.maxvirtualdim(ψ) == 1
        @test TN.scalartype(ψ) == eltype
        @test all([length(inds(ψ[v])) == degree(g, v) + 1 for v in vertices(ψ)])
    end

end

end