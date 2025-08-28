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

function main(nx::Int64, ny::Int64, nz::Int64, χ::Int64, boundary::String, annealing_time::Int64, disorder_no::Int64)
    println("Beginning simulation with chi = " *string(χ)* " an annealing time of "*string(annealing_time)* " a system size of " *string(nx)*string(ny)*string(nz) * " and a boundary of "*boundary *" and a disorder of "*string(disorder_no))
    cutoff = 1e-12

    disorder_no_str = string(disorder_no - 1, pad = 2)
    instance_file = boundary == "obc" ? "/mnt/home/jtindall/ceph/Data/DWave/Instances/obcdiamond_($(nx), $(ny), $(nz))_precision256/seed"*disorder_no_str*".npz" : "/mnt/home/jtindall/ceph/Data/DWave/Instances/diamond_($(nx), $(ny), $(nz))_precision256/seed"*disorder_no_str*".npz"
    g, J_dict = graph_couplings_from_instance(instance_file)
    h_dict = Dictionary([v for v in vertices(g)], [1 for v in vertices(g)])
    k = maximum([degree(g, v) for v in vertices(g)])
    ecs = edge_color(g, k)
    ordered_edges = reduce(vcat, ecs)

    Random.seed!(1234)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "X-", s)
    dβs = [(10, 0.25), (100, 0.1), (200, 0.01), (200, 0.005), (200, 0.001)]

    δt = 0.01
    nsteps = Int64(annealing_time / δt) + 1

    χ_GS = 4

    apply_kwargs = (maxdim = χ, cutoff, normalize = true)
    gs_apply_kwargs = (maxdim = χ_GS, cutoff, normalize = true)
    ψIψ = build_bp_cache(ψ)

    t = 0

    println("Performing imaginary time evo.")
    Gamma_s_init, J_s_init = annealing_schedule(t, annealing_time)
    for (nGSsteps, dβ) in dβs
        layer = []
        append!(layer, ("Rx", [v], -1.0 * im * dβ * Gamma_s_init * h_dict[v]) for v in vertices(g))
        append!(layer, ("Rzz", pair, -2.0 * J_s_init * im * dβ * J_dict[pair]) for pair in ordered_edges)
        append!(layer, ("Rx", [v], -1.0 * im * dβ * Gamma_s_init * h_dict[v]) for v in vertices(g))
        for i in 1:nGSsteps
            ψ, ψIψ, errs = apply(layer, ψ, ψIψ; apply_kwargs = gs_apply_kwargs)
        end
    end

    s = siteinds(ψ)
    ψIψ = build_bp_cache(ψ)
    ψ, ψIψ  = normalize(ψ, ψIψ; update_cache = false)

    sf = 1.0 * pi

    t_start = time()

    s_cutoff = 0.6

    all_errs = []
    flush(stdout)
    t1 = time()
    for i in 1:round(Int, s_cutoff*nsteps)
        sr = t / annealing_time
        println("Time is $t (ns), s is $sr")
        Gamma_s, J_s = annealing_schedule(t + 0.5*δt, annealing_time)
        @show (Gamma_s, J_s)

        Gamma_s, J_s = sf * Gamma_s, sf * J_s

        layer = []
        append!(layer, ("Rx", [v], δt * Gamma_s * h_dict[v]) for v in vertices(g))
        append!(layer, ("Rzz", pair, 2 * J_s * δt * J_dict[pair]) for pair in ordered_edges)
        append!(layer, ("Rx", [v], δt * Gamma_s * h_dict[v]) for v in vertices(g))

        ψ, ψIψ, errs = apply(layer, ψ, ψIψ; apply_kwargs)
        errs = errs[(length(vertices(g))+1):(length(vertices(g))+1+length(ordered_edges))]
        all_errs = push!(all_errs, errs)
        
        max_linkdim=  maxlinkdim(ψ)
        println("Maximum bond dimension is $max_linkdim")
        t += δt

        flush(stdout)
    end
    t2 = time()

    time_taken = t2 - t1
    npzwrite("/mnt/home/jtindall/ceph/Data/DWave/PaperData/ResubmissionData/SMErrorAnalysisFigure/Timings/nx$(nx)ny$(ny)nz$(nz)Chi$(χ)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).npz", time_taken = time_taken)
end

ns = [(3,3,8), (4,4,8), (5,5,8), (8,8,8), (8,8,12), (8,8,16), (10,10,12), (10,10,16), (12,12,16), (10,10,24), (12,12,24)]
disorder_no = parse(Int64, ARGS[1])
annealing_time = parse(Int64, ARGS[2])
χ = parse(Int64, ARGS[3])
bc = "pbc"

for (nx, ny, nz) in ns
    main(nx, ny, nz, χ, bc, annealing_time, disorder_no)
end