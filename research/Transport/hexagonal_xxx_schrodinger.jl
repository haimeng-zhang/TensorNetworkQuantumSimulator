using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices

using Random
using TOML

include("utils.jl")

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()


function main()

    ny = 4
    g = named_hexagonal_cylinder(ny)
    s = siteinds("S=1/2", g)

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])
    ψ = ITN.ITensorNetwork(v -> first(v) <= column_lengths[last(v)] / 2 ? "Z+" : "Z-", s)

    ψψ = TN.build_bp_cache(ψ)
    obs = [("Z", [v]) for v in collect(vertices(ψ))]

    init_mag = sum(TN.expect(ψψ, obs))
    println("Initial magnetisation is $init_mag")

    δt = 0.1

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 3)
    for colored_edges in ec
        append!(layer, ("Rxxyyzz", pair, 2*δt) for pair in colored_edges)
    end

    no_trotter_steps = 2
    measure_freq = 1

    t = 0
    apply_kwargs = (; maxdim = 4, cutoff = 1e-10)
    for i in 1:no_trotter_steps
        ψ, ψψ, errs = apply(layer, ψ, ψψ; apply_kwargs)

        if i % measure_freq == 0
            println("Time is $(t)")
            bp_mags = TN.expect(ψψ, obs)
            println("Maximum bond dimension is $(ITN.maxlinkdim(ψ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")
            println("Total BP magnetisation is $(sum(bp_mags))")

            ψψ_bmps = build_boundarymps_cache(ψ, 8; cache_update_kwargs = (; maxiter = 5))
            bmps_mags = TN.expect(ψψ_bmps, obs)
            @show sum(abs.(bp_mags - bmps_mags))
            println("Total BMPS magnetisation is $(sum(bmps_mags))")
        end
        t += δt
    end
end

main()
