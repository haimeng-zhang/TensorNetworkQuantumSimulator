@eval module $(gensym())
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
using Test: @testset

@testset "Test examples" begin
    example_files = [
        "boundarymps.jl", "heavyhex_isingdynamics.jl", "loopcorrections.jl", "TEBD.jl", "time_evolution.jl", "time_evolution_Heisenberg.jl",
    ]
    @testset "Test $example_file" for example_file in example_files
        include(joinpath(pkgdir(TensorNetworkQuantumSimulator), "examples", example_file))
    end
end
end
