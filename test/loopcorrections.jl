using TensorNetworkQuantumSimulator
import TensorNetworkQuantumSimulator as TN

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using LinearAlgebra: norm

using EinExprs: Greedy

using Random
Random.seed!(1634)

@testset "Loop Corrections" begin
    nx, ny = 4, 4
    χ = 3
    ITensors.disable_warn_order()
    gs = [
        (named_grid((nx, 1)), "line", 0),
        (named_hexagonal_lattice_graph(nx, ny), "hexagonal", 6),
        (named_grid((nx, ny)), "square", 4),
    ]
    for (g, g_str, smallest_loop_size) in gs
        ψ = TN.random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ)

        ψ = normalize(ψ; alg = "bp")

        norm_bp = norm(ψ; alg = "bp")
        norm_loopcorrected = norm(ψ; alg = "loopcorrections", max_configuration_size = 2 * (smallest_loop_size) - 1)
        norm_exact = norm(ψ; alg = "exact")

        @test isapprox(norm_bp, norm_exact, atol = 5e-2)
        @test isapprox(norm_loopcorrected, norm_exact, atol = 5e-2)
    end
end