using TensorNetworkQuantumSimulator
using Documenter

DocMeta.setdocmeta!(TensorNetworkQuantumSimulator, :DocTestSetup, :(using TensorNetworkQuantumSimulator); recursive = true)

makedocs(;
    modules = [TensorNetworkQuantumSimulator],
    authors = ["JoeyT1994 <jtindall@flatironinstitute.org>", "MSRudolph <manuel.rudolph@web.de>", "Xuanzhao Gao <xgao@flatironinstitute.org>", "and contributors"],
    sitename = "TensorNetworkQuantumSimulator.jl",
    format = Documenter.HTML(;
        canonical = "https://JoeyT1994.github.io/TensorNetworkQuantumSimulator.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl",
    devbranch = "main",
)
