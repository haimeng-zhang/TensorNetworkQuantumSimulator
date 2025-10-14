using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics
using NamedGraphs: AbstractNamedGraph

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

using StatsBase

using Dictionaries: Dictionary, set!

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

IT.disable_warn_order()

function get_job_corrs(g::AbstractNamedGraph, job_no::Int64, no_corrs_per_job::Int64)
    job_v1_v2s = []
    vs = collect(vertices(g))
    nvs = length(vs)
    count = 0
    for (i, v1) in enumerate(1:nvs)
        for (j, v2) in enumerate((i+1):nvs)
            if count >= (job_no - 1)*no_corrs_per_job
                push!(job_v1_v2s, (v1, v2))
            end
            count += 1
            length(job_v1_v2s) == no_corrs_per_job && return job_v1_v2s
        end
    end
    return job_v1_v2s
end

function which_corr(v1, v2, nvs)
    vmax, vmin = maximum((v1, v2)), minimum((v1,v2))
    return sum([nvs - v for v in 1:(vmin-1)]) + (vmax-vmin)
end

    
function main_diamond()
    println("Diamond Regular analysis")
    nx, ny, nz = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3])
    disorder_no = parse(Int64, ARGS[4])
    annealing_time = parse(Int64, ARGS[5])
    no_corrs = parse(Int64, ARGS[6])
    instance = parse(Int64, ARGS[7])
    χ_state = 16
    max_loop_length = 6


    f = nothing
    try
        f= load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ_state)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2")
    catch
        f= load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ_state)DisorderNo$(disorder_no)AnnealingTime$(annealing_time)Cutoff.jld2")
    end
    ψ = f["Wavefunction"]

    ψIψ = build_normsqr_bp_cache(ψ)
    vs = collect(vertices(ψ))
    Random.seed!(instance*123)
    v1v2s = [(v1, v2) for (i,v1) in enumerate(vs) for v2 in vs[i+1:length(vs)]]
    v1v2s = StatsBase.sample(v1v2s, no_corrs, replace = false)
    @show v1v2s
    egs = NG.edgeinduced_subgraphs_no_leaves(ITN.partitioned_graph(ψIψ), max_loop_length)
    loopcorrected_corrs = ComplexF64[]
    bp_corrs= ComplexF64[]

    flush(stdout)

    for (i, (v1, v2)) in enumerate(v1v2s)
        println("Computing Correlation No. $i of $(length(v1v2s))")
        corrs = zz_correlation_bp_loopcorrectfull(ψ, v1, v2, egs)
        push!(loopcorrected_corrs, last(corrs))
        push!(bp_corrs, first(corrs))
        flush(stdout)
    end

    npzwrite("/mnt/home/jtindall/ceph/Data/DWave/PaperData/ResubmissionData/DiamondKibbleZurek/$(no_corrs)Corrsnx$(nx)ny$(ny)nz$(nz)AnnealingTime$(annealing_time)Chi16DisorderNo$(disorder_no)Instance$(instance).npz",
            bp_corrs = bp_corrs, loopcorrected_corrs = loopcorrected_corrs)
end

main_diamond()
