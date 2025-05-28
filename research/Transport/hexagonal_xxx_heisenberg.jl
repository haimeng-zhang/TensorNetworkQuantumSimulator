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

function high_temperature_initial_state(sphysical, sancilla, mu, vertex_lower_half_filter)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = vertex_lower_half_filter(v) ? (1/2)*[1 + mu 0; 0 1 - mu] : (1/2)*[1 - mu 0; 0 1 + mu]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end

function identity_state(sphysical, sancilla)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = [1 0; 0 1]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end

function sigmaz_state(sphysical, sancilla, vz)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = v == vz ? [1 0; 0 -1] : [1 0; 0 1]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end


function main()

    ny = 12
    g = named_hexagonal_cylinder(ny)
    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)
    mu = 0.1

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])
    ρ = high_temperature_initial_state(sphysical, sancilla, mu, v -> first(v) <= column_lengths[last(v)] / 2)
    ρρ = build_bp_cache(ρ)
    ρId = identity_state(sphysical, sancilla)
    println("Intial trace is $(inner(ρ, ρId; alg = "bp"))")


    obs = [("Z", [v]) for v in collect(vertices(g))]

    init_mag = sum([inner(ρ, sigmaz_state(sphysical, sancilla, v); alg = "bp") for v in vertices(g)])
    println("Initial magnetisation is $init_mag")

    δt = 0.1

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 3)
    for colored_edges in ec
        _layer = reduce(vcat, [[ITensors.op("Rxxyyzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θ = 2*δt), ITensors.op("Rxxyyzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θ = -2*δt)] for pair in colored_edges])
        append!(layer, _layer)
    end

    no_trotter_steps = 500
    measure_freq = 1

    t = 0
    f = "/mnt/home/jtindall/ceph/Data/Transport/Hexagonal/HeisenbergPicture/ny"*string(ny)*"maxdim"*string(maxdim)*"deltat"*string(δt)*"mu"*string(mu)
    apply_kwargs = (; maxdim = 64, cutoff = 1e-8, normalize = false)
    for i in 1:no_trotter_steps
        ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs)

        if i % measure_freq == 0
            println("Time is $(t)")
            println("Maximum bond dimension is $(ITN.maxlinkdim(ρ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")
            tr = inner(ρ, ρId; alg = "bp")
            println("Trace is $(tr)")

            bp_mags = [inner(ρ, sigmaz_state(sphysical, sancilla, v); alg = "bp") for v in vertices(g)]
            file_name = f * "TrotterStep"*string(i)*".npz"
            println("Current magnetisation is $(sum(bp_mags))")
            npzwrite(file_name, trace =tr, bp_mags = bp_mags, rows = first.(collect(vertices(g))), cols = last.(collect(vertices(g))))
        end
        t += δt
    end
end

main()
