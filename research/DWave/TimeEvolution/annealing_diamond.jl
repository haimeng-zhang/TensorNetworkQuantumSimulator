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

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
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

    apply_kwargs = (maxdim = χ, cutoff, normalize_tensors = true)
    gs_apply_kwargs = (maxdim = χ_GS, cutoff, normalize_tensors = true)
    ψIψ = build_normsqr_bp_cache(ψ)

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
    ψIψ = build_normsqr_bp_cache(ψ)
    ψ, ψIψ  = normalize(ψ, ψIψ; update_cache = false)

    sf = 1.0 * pi

    t_start = time()

    s_cutoff = 0.6

    all_errs = []
    flush(stdout)
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

    all_errs = reduce(vcat, all_errs)
    ψ, ψIψ  = normalize(ψ, ψIψ; update_cache = false)
    loop_errors = TN.loop_correlations(ψIψ, 6)
    ψ = ITN.VidalITensorNetwork(ψ; cache! = Ref(ψIψ), cache_update_kwargs = (; maxiter = 0))
    ψ = ITensorNetwork(ψ)
    total_time = time() - t_start
    if boundary == "obc"
        save("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/OBCDiamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2", Dict("Wavefunction" => ψ, "TimeTaken" => total_time, "Errors" => all_errs, "LoopErrors" => loop_errors))
    elseif boundary == "pbc"
        save("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2", Dict("Wavefunction" => ψ, "TimeTaken" => total_time, "Errors" => all_errs, "LoopErrors" => loop_errors))
    end
end

nx, ny, nz = parse(Int64, ARGS[1]),  parse(Int64, ARGS[2]),  parse(Int64, ARGS[3])
disorder_no = parse(Int64, ARGS[4])
annealing_time = parse(Int64, ARGS[5])
χ = parse(Int64, ARGS[6])
bc = ARGS[7]

main(nx, ny, nz, χ, bc, annealing_time, disorder_no)
