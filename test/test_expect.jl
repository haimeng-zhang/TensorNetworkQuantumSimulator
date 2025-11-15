@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: datatype
using Graphs: center, is_tree, neighbors
using NamedGraphs: edges, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_hexagonal_lattice_graph
using Random
using Statistics
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Expect" begin
    nx, ny = 4, 4
    χ = 2

    gs = [
        (named_grid((nx, 1)), "line"),
        (named_hexagonal_lattice_graph(nx - 2, ny - 2), "hexagonal"),
        (named_grid((nx, ny)), "square"),
    ]
    for (g, g_str) in gs
        ψ = TN.random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ)
        v_centre = first(center(g))

        sz_exact = TN.expect(ψ, ("Z", [v_centre]); alg = "exact")
        sz_bp = TN.expect(ψ, ("Z", [v_centre]); alg = "bp")

        if is_tree(g)
            @test sz_bp ≈ sz_exact
        else
            @test sz_bp != sz_exact
        end

        Rmps = 16
        sz_boundarymps = TN.expect(ψ, ("Z", [v_centre]); alg = "boundarymps", mps_bond_dimension = Rmps)

        @test sz_boundarymps ≈ sz_exact

        if !is_tree(g)
            v_centre_neighbor = first(neighbors(g, v_centre))
            sz_exact = TN.expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "exact")
            sz_boundarymps = TN.expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "boundarymps", mps_bond_dimension = Rmps)

            @test sz_boundarymps ≈ sz_exact
        end
    end
end

end
