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


function main(seed::Int, χ::Int, ny::Int, mu::Float64, δt::Float64, Δ::Float64)

    g, column_lengths = named_heavy_hexagonal_cylinder(ny)
    s = ITensorNetworks.siteinds("S=1/2", g)

    Random.seed!(1234*seed)
    function up_down(v)
        if first(v) <= column_lengths[last(v)] / 2
            rand() < 0.5*(1 + mu) && return "Z+"
            return "Z-"
        else
            rand() < 0.5*(1 - mu) && return "Z+"
            return "Z-"
        end
    end
    ψ = ITN.ITensorNetwork(v -> up_down(v), s)

    ψψ = TN.build_bp_cache(ψ)
    obs = [("Z", [v]) for v in collect(vertices(ψ))]

    init_mags = ComplexF64[o for o in TN.expect(ψψ, obs)]
    println("Initial magnetisation is $(sum(init_mags))")

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 3)
    for colored_edges in ec
        _layer =[ITensors.op("RxxyyRzz", only(s[src(pair)]), only(s[dst(pair)]); θxy = 2*δt, θz = 2*Δ*δt) for pair in colored_edges]
        append!(layer, _layer)
    end

    no_trotter_steps = 100
    measure_freq = 1

    t = 0
    apply_kwargs = (; maxdim = χ, cutoff = 1e-10)
    f = "/mnt/home/jtindall/ceph/Data/Transport/HeavyHexagonal/SchrodingerPicture/Seed"*string(seed)*"ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"Delta"*string(Δ)

    rows = Int64[r for r in first.(collect(vertices(g)))]
    cols = Int64[r for r in last.(collect(vertices(g)))]
    file_name = f * "TrotterStep0.npz"
    npzwrite(file_name, bp_mags = init_mags, bmps_mags = init_mags, rows = rows, cols = cols)

    for i in 1:no_trotter_steps
        ψ, ψψ, errs = apply(layer, ψ, ψψ; apply_kwargs)

        if i % measure_freq == 0
            println("Time is $(t)")
            bp_mags = TN.expect(ψψ, obs)
            println("Maximum bond dimension is $(ITN.maxlinkdim(ψ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")
            println("Total BP magnetisation is $(sum(bp_mags))")

            #ψψ_bmps = build_boundarymps_cache(ψ, 2*ITensorNetworks.maxlinkdim(ψ); cache_update_kwargs = (; maxiter = 5))
            #bmps_mags = TN.expect(ψψ_bmps, obs)
            bmps_mags = [0 for m in bp_mags]
            println("Total BMPS magnetisation is $(sum(bmps_mags))")

            diffs = sum(abs.(bmps_mags - bp_mags)) / length(bp_mags)
            println("Average diff between bp and mps is $(sum(diffs))")
            bp_mags = ComplexF64[b for b in bp_mags]
            bmps_mags = ComplexF64[b for b in bmps_mags]
            file_name = f * "TrotterStep"*string(i)*".npz"
            npzwrite(file_name, bp_mags = bp_mags, bmps_mags = bmps_mags, rows = rows, cols = cols)
        end
        t += δt
    end
end

#seed, χ, ny, mu, δt, Δ = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3]), parse(Float64, ARGS[4]), parse(Float64, ARGS[5]), parse(Float64, ARGS[6])
seed, χ, ny, mu, δt, Δ = 1, 150, 30, 1.0, 0.1, 1.0
main(seed, χ, ny, mu, δt, Δ)
