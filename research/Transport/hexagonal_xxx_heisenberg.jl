using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using ITensorNetworks: AbstractBeliefPropagationCache
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

function trace_expect(ρI::AbstractBeliefPropagationCache, obs::Vector{<:Tuple}, sphysical, sancilla)
    os = []
    for ob in obs
        op_strs, vs, coeff = TN.collectobservable(ob)
        incoming_messages = ITensorNetworks.environment(ρI, [(v, "bra") for v in vs])
        local_numer_ops = [ITensors.replaceind(ITensors.op(op_str, only(sphysical[v])), prime(only(sphysical[v])), only(sancilla[v]))  for (op_str, v) in zip(op_strs, vs)]
        ts = [incoming_messages; local_numer_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        numer = coeff * ITensors.contract(ts; sequence = seq)[]

        local_denom_ops = [ρI[(v, "bra")]  for v in vs]
        ts = [incoming_messages; local_denom_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        denom = ITensors.contract(ts; sequence = seq)[]
        push!(os, numer / denom)
    end
    return os
end

function trace_expect(tr_ρ::AbstractBeliefPropagationCache, ρ::ITensorNetwork, obs::Vector{<:Tuple}, sphysical, sancilla)
    os = []
    for ob in obs
        op_strs, vs, coeff = TN.collectobservable(ob)
        incoming_messages = ITensorNetworks.environment(tr_ρ, vs)
        local_numer_ops = [ITensors.replaceind(ITensors.op(op_str, only(sphysical[v])), prime(only(sphysical[v])), only(sancilla[v]))  for (op_str, v) in zip(op_strs, vs)]
        local_numer_ops = [local_numer_ops[i] * ρ[v] for (i, v) in enumerate(vs)]
        ts = [incoming_messages; local_numer_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        numer = coeff * ITensors.contract(ts; sequence = seq)[]

        local_denom_ops = [ρ[v] * delta(only(sphysical[v]), only(sancilla[v])) for (i, v) in enumerate(vs)]
        ts = [incoming_messages; local_denom_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        denom = ITensors.contract(ts; sequence = seq)[]
        push!(os, numer / denom)
    end
    return os
end

function form_tr_ρ(ρ::ITensorNetwork, sphysical, sancilla)
    tr_ρ = copy(ρ)
    for v in vertices(ρ)
        ITensorNetworks.@preserve_graph tr_ρ[v] = ρ[v] * delta(only(sphysical[v]), only(sancilla[v]))
    end
    return tr_ρ
end

function main()

    ny = 4
    g = named_hexagonal_cylinder(ny)
    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)
    mu = 0.1

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])
    ρ = high_temperature_initial_state(sphysical, sancilla, mu, v -> first(v) <= column_lengths[last(v)] / 2)
    ρρ = build_bp_cache(ρ)
    tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
    tr_ρ = ITensorNetworks.BeliefPropagationCache(tr_ρ)
    tr_ρ = TN.updatecache(tr_ρ)
    println("Intial trace is $(scalar(tr_ρ))")


    obs = [("Z", [v]) for v in collect(vertices(g))]

    init_mags = trace_expect(tr_ρ, ρ, obs, sphysical, sancilla)
    println("Initial magnetisation is $(sum(init_mags))")

    δt = 0.1

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 3)
    for colored_edges in ec
        _layer = reduce(vcat, [[ITensors.op("Rxxyyzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θ = 2*δt), ITensors.op("Rxxyyzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θ = -2*δt)] for pair in colored_edges])
        append!(layer, _layer)
    end

    no_trotter_steps = 100
    measure_freq = 1

    t = 0
    f = "/mnt/home/jtindall/ceph/Data/Transport/Hexagonal/HeisenbergPicture/ny"*string(ny)*"maxdim"*string(maxdim)*"deltat"*string(δt)*"mu"*string(mu)
    apply_kwargs = (; maxdim = 32, cutoff = 1e-8, normalize = false)
    for i in 1:no_trotter_steps
        ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs)

        if i % measure_freq == 0
            println("Time is $(t)")
            println("Maximum bond dimension is $(ITN.maxlinkdim(ρ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")

            tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
            tr_ρ = TN.updatecache(ITensorNetworks.BeliefPropagationCache(tr_ρ))
            println("Trace is $(scalar(tr_ρ))")


            bp_mags = trace_expect(tr_ρ,ρ, obs, sphysical, sancilla)
            file_name = f * "TrotterStep"*string(i)*".npz"
            println("Current BP Measured magnetisation is $(sum(bp_mags))")

            tr_ρ_bmps = TN.BoundaryMPSCache(ITensorNetworks.BeliefPropagationCache(tr_ρ); message_rank = 2*ITensorNetworks.maxlinkdim(ρ),
            grouping_function = v -> last(v), group_sorting_function = v -> first(v))
            tr_ρ_bmps = TN.updatecache(tr_ρ_bmps; alg = "orthogonal", maxiter = 5, message_update_kwargs = (; niters = 50, tolerance = 1e-10))
            bmps_mags = trace_expect(tr_ρ_bmps, ρ, obs, sphysical, sancilla)
            println("Current BMPS Measured magnetisation is $(sum(bmps_mags))")

            diffs = sum(abs.(bmps_mags - bp_mags)) / length(bp_mags)
            println("Average diff between bp and mps is $(sum(diffs))")
            #npzwrite(file_name, trace =tr, bp_mags = bp_mags, rows = first.(collect(vertices(g))), cols = last.(collect(vertices(g))))
        end
        t += δt
    end
end

main()
