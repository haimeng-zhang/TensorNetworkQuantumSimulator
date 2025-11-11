using TensorNetworkQuantumSimulator
using Test

@testset "TensorNetworkQuantumSimulator.jl" begin
    include("test_constructors.jl")
    include("test_forms.jl")
    include("test_examples.jl")
end
