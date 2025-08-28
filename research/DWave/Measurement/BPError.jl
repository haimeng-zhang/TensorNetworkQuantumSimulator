using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

include("../utils.jl")

using Random
using Serialization

using JLD2

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using ITensors
const IT = ITensors

using Dictionaries: Dictionary, set!

BLAS.set_num_threads(min(8, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

IT.disable_warn_order()

function main(nx::Int64, ny::Int64, nz::Int64, χ::Int64, annealing_time::Int64, disorder_no::Int64)
    ψ = load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2")["Wavefunction"]
    ψψ = build_bp_cache(ψ)
    errs = TN.loop_correlations(ψψ, 6)
    @show Statistics.mean(errs)
    @show Statistics.maximum(errs)
    @show Statistics.median(errs)

    #@show reverse(sort(errs))
end

nx, ny, nz = 8, 8, 12
χ = 8
annealing_time = 7
disorder_no = 1

main(nx, ny, nz, χ, annealing_time, disorder_no)