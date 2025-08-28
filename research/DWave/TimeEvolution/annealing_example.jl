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

using ITensors: ITensors, Algorithm
const IT = ITensors

using Dictionaries: Dictionary, set!

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

IT.disable_warn_order()

function main()
    cutoff = 1e-12

    #Set the annelaing time and maximum bond dimension of the state
    annealing_time = 1
    χ = 4

    #Define the lattice (here is 1D chain)
    #g = named_grid((4,1); periodic = true)
    g = named_cylinder(5,5)

    #Form a dictionary mapping the edges of the graph to couplings J_{i,j}
    J_dict = Dictionary(edges(g), [Random.rand() for e in edges(g)])
    #Form a dictionary mapping the vertices of the graph to field strengths
    h_dict = Dictionary([v for v in vertices(g)], [1 for v in vertices(g)])

    #Figure out an edge coloring to Trotterise correctly
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

    println("Performing imaginary time evo to get initial state.")
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

    ψ, ψIψ  = normalize(ψ, ψIψ; update_cache = false)

    sf = 1.0 * pi

    #Use a cutoff as the dynamics becomes fairly trivial after this point
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

    #Now we measure and we need to do things carefully and in different ways depending on the lattice

    #Lets measure "ZZ" on two given vertices
    v1 = (2, 1)
    v2 = (2, 4)

    #If the graph is a tree (/ we want to make the vanilla BP approximation) the following will suffice
    if NamedGraphs.is_tree(ψ)
        #This will be empty 
        egs = NG.edgeinduced_subgraphs_no_leaves(ITN.partitioned_graph(ψIψ), 0)

        zz_expect = only(zz_correlation_bp_loopcorrectfull(ψ, v1, v2, egs))
        println("ZZ expectation on sites $(v1) and $(v2) is $(zz_expect)")

        #Note that local expectation values under the BP approximation can be obtained via the usual
        #local_expect = expect(ψ, ("X", [v1]); alg = "bp")
    else
        #Define a maximum loop size to correct up to, keep lmax <= (2 * smallest_size - 1)
        lmax = 7
 
        egs = NG.edgeinduced_subgraphs_no_leaves(ITN.partitioned_graph(ψIψ), lmax)

        #This will return a vector of the zz corr computed with increased loop rank lmax up to lmax
        zz_expect = zz_correlation_bp_loopcorrectfull(ψ, v1, v2, egs)

        for (i, circuit_length) in enumerate(vcat([0], sort(unique(length.(edges.(egs))))))
            println("Loop Corrected ZZ expectation with lmax = $(circuit_length) on sites $(v1) and $(v2) is $((zz_expect)[i])")
        end
    end

    #Expert mode, boundary MPS to measure things if the lattice is cylindrical or planar lattice. On a cylinder, this will be very accurate when the circumference of the cylinder is large and it converges in boundarymps_dim
    #On a planar lattice this will be very accurate when it converges in boundarymps_dim
    boundarymps_dim = 10
    #If a cylinder use maxiter ~ 5, if planar (no periodic boundary), use maxiter ~ 1
    maxiter = 5
    ψIψ_bmps = TN.build_normsqr_bmps_cache(copy(ψIψ),boundarymps_dim; cache_update_kwargs= (; maxiter, message_update_alg = Algorithm("orthogonal", niters =20)))
    #Any column-aligned expectation value is allowed here (i.e. any number of pauli strings, the vertices just have to belong to the same column). Non column-aligned requires some expert mode functions hidden in utils.jl
    zz_expect = expect(ψIψ_bmps, ("ZZ", [v1, v2]))
    println("ZZ expectation on sites $(v1) and $(v2) via boundary mps with mps dimension $(boundarymps_dim) is $(zz_expect)")

end

main()