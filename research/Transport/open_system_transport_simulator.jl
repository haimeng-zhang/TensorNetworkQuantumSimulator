using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using ITensorNetworks: AbstractBeliefPropagationCache, IndsNetwork
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

function sqrt_high_temperature_initial_state(sphysical, sancilla, mu, vertex_lower_half_filter)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = vertex_lower_half_filter(v) ? (1/sqrt(2))*[sqrt(1 + mu) 0; 0 sqrt(1 - mu)] : (1/sqrt(2))*[sqrt(1 - mu) 0; 0 sqrt(1 + mu)]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end

function identity_state(sphysical, sancilla; normalize = false)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = [1 0; 0 1]
        λ = normalize ? 0.5 : 1 
        ITensorNetworks.@preserve_graph ψ[v] = λ * ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
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

        local_denom_ops = [ρ[v] * ITensors.delta(only(sphysical[v]), only(sancilla[v])) for (i, v) in enumerate(vs)]
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
        ITensorNetworks.@preserve_graph tr_ρ[v] = ρ[v] * ITensors.delta(only(sphysical[v]), only(sancilla[v]))
    end
    return tr_ρ
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obs::Tuple)
    op_vec, vs, coeff = TN.collectobservable(obs)

    ρOρ = copy(ρρ)
    for (i, v) in enumerate(vs)
        ITensorNetworks.@preserve_graph ρOρ[(v,"operator")] = ITensors.op("Id", only(sancilla[v])) * ITensors.op(op_vec[i], only(sphysical[v]))
    end

    numerator = ITensorNetworks.region_scalar(ρOρ, [(v, "ket") for v in vs])
    denominator = ITensorNetworks.region_scalar(ρρ, [(v, "ket") for v in vs])

    return coeff * numerator / denominator
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obss::Vector{<:Tuple})
    return [expect(ρρ, sphysical, sancilla, obs) for obs in obss]
end



function main_chain()

    Random.seed!(12*seed)

    n = 5
    g, bottom_half_vertices = transport_graph_constructor("Chain", n)
    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)

    ρ = identity_state(sphysical, sancilla)
    ρρ = build_bp_cache(ρ)
    println("Intial trace is $(scalar(ρρ))")

    @show ρ

    tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
    tr_ρ = ITensorNetworks.BeliefPropagationCache(tr_ρ)
    tr_ρ = TN.updatecache(tr_ρ; maxiter = 50, tol = 1e-10)
    println("Intial trace is $(scalar(tr_ρ))")

    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]


    obs = [("Z", [v]) for v in collect(vertices(g))]

    init_mags = ComplexF64[o for o in trace_expect(tr_ρ, ρ, obs, sphysical, sancilla)]
    println("Initial magnetisation is $(sum(init_mags))")

    δt = 0.01
    Δ = 1.0
    nsteps = 5000


    lattice = "Chain"
    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    ec = edge_color(g, k)
    for colored_edges in ec
        _layer = reduce(vcat, [[ITensors.op("RxxyyRzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θxy = -2*δt, θz = -2*Δ*δt), ITensors.op("RxxyyRzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θxy = 2*δt, θz = 2*Δ*δt)] for pair in colored_edges])
        append!(layer, _layer)
    end

    vp, vm = (1,1), (n,1)
    γp, γm = 1.4, 1.0

    jump_ops = [γp * ITensors.op("S+", only(sphysical[vp]))* ITensors.op("S+", only(sancilla[vp])) - 0.5 * γp * ITensors.op("ProjDn", only(sphysical[vp])) * ITensors.op("Id", only(sancilla[vp])) - 0.5 * γp * ITensors.op("ProjDn", only(sancilla[vp])) * ITensors.op("Id", only(sphysical[vp])),
    γm * ITensors.op("S-", only(sphysical[vm]))* ITensors.op("S-", only(sancilla[vm])) - 0.5 * γm * ITensors.op("ProjUp", only(sphysical[vm])) * ITensors.op("Id", only(sancilla[vm])) - 0.5 * γm * ITensors.op("ProjUp", only(sancilla[vm])) * ITensors.op("Id", only(sphysical[vm]))]
    jump_layer = [exp(0.5 * δt * o) for o in jump_ops]

    χ = 32
    apply_kwargs = (; maxdim = χ, cutoff = 1e-12, normalize = false)

    for i in 1:nsteps
        ρ, ρρ, errs = apply(jump_layer, ρ, ρρ; apply_kwargs, update_cache = false)
        ρρ = updatecache(ρρ)
        ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs)
        ρ, ρρ, errs = apply(jump_layer, ρ, ρρ; apply_kwargs, update_cache = false)
        ρρ = updatecache(ρρ)


        @show mean(errs)
        tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
        tr_ρ = ITensorNetworks.BeliefPropagationCache(tr_ρ)
        tr_ρ = TN.updatecache(tr_ρ; maxiter = 50, tol = 1e-10)
        println("Trace is $(scalar(tr_ρ))")

        mags = ComplexF64[o for o in trace_expect(tr_ρ, ρ, obs, sphysical, sancilla)]
        println("Total magnetisation is $(sum(mags))")

        @show mags
    end

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

        local_denom_ops = [ρ[v] * ITensors.delta(only(sphysical[v]), only(sancilla[v])) for (i, v) in enumerate(vs)]
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
        ITensorNetworks.@preserve_graph tr_ρ[v] = ρ[v] * ITensors.delta(only(sphysical[v]), only(sancilla[v]))
    end
    return tr_ρ
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obs::Tuple)
    op_vec, vs, coeff = TN.collectobservable(obs)

    ρOρ = copy(ρρ)
    for (i, v) in enumerate(vs)
        ITensorNetworks.@preserve_graph ρOρ[(v,"operator")] = ITensors.op("Id", only(sancilla[v])) * ITensors.op(op_vec[i], only(sphysical[v]))
    end

    numerator = ITensorNetworks.region_scalar(ρOρ, [(v, "ket") for v in vs])
    denominator = ITensorNetworks.region_scalar(ρρ, [(v, "ket") for v in vs])

    return coeff * numerator / denominator
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obss::Vector{<:Tuple})
    return [expect(ρρ, sphysical, sancilla, obs) for obs in obss]
end



function main_hexagonal()

    Random.seed!(12*seed)

    n = 1
    g, _ = transport_graph_constructor("Hexagonal", n)

    g = named_hexagonal_lattice_graph(1,1)
    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)

    ρ = identity_state(sphysical, sancilla; normalize = true)
    ρρ = build_bp_cache(ρ)

    tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
    tr_ρ = ITensorNetworks.BeliefPropagationCache(tr_ρ)
    tr_ρ = TN.updatecache(tr_ρ; maxiter = 50, tol = 1e-10)
    println("Intial trace is $(scalar(tr_ρ))")


    obs = [("Z", [v]) for v in collect(vertices(g))]

    init_mags = ComplexF64[o for o in trace_expect(tr_ρ, ρ, obs, sphysical, sancilla)]
    println("Initial magnetisation is $(sum(init_mags))")

    δt = 0.01
    Δ = 1.0
    nsteps = 1000


    lattice = "Hexagonal"
    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    ec = edge_color(g, k)
    for colored_edges in ec
        _layer = reduce(vcat, [[ITensors.op("RxxyyRzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θxy = -2*δt, θz = -2*Δ*δt), ITensors.op("RxxyyRzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θxy = 2*δt, θz = 2*Δ*δt)] for pair in colored_edges])
        append!(layer, _layer)
    end

    column_lengths = maximum.([first.(filter(v -> last(v) == i, collect(vertices(g)))) for i in unique(last.(vertices(g)))])
    vsources = filter(v -> first(v) == 1, collect(vertices(g)))
    vsinks =  filter(v -> first(v) == column_lengths[last(v)], collect(vertices(g)))
    γp, γm = 2.0, 1.0

    @show vsources
    @show vsinks

    source_jump_ops = [γp * ITensors.op("S+", only(sphysical[vp]))* ITensors.op("S+", only(sancilla[vp])) - 0.5 * γp * ITensors.op("ProjDn", only(sphysical[vp])) * ITensors.op("Id", only(sancilla[vp])) - 0.5 * γp * ITensors.op("ProjDn", only(sancilla[vp])) * ITensors.op("Id", only(sphysical[vp])) for vp in vsources]
    sink_jump_ops = [γm * ITensors.op("S-", only(sphysical[vm]))* ITensors.op("S-", only(sancilla[vm])) - 0.5 * γm * ITensors.op("ProjUp", only(sphysical[vm])) * ITensors.op("Id", only(sancilla[vm])) - 0.5 * γm * ITensors.op("ProjUp", only(sancilla[vm])) * ITensors.op("Id", only(sphysical[vm])) for vm in vsinks]
    jump_layer = [exp(0.5 * δt * o) for o in vcat(source_jump_ops, sink_jump_ops)]

    χ = 16
    apply_kwargs = (; maxdim = χ, cutoff = 1e-12, normalize = false)

    for i in 1:nsteps
        ρ, ρρ, errs = apply(jump_layer, ρ, ρρ; apply_kwargs, update_cache = false)
        ρρ = updatecache(ρρ)
        ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs)

        @show mean(errs)

        ρ, ρρ, errs = apply(jump_layer, ρ, ρρ; apply_kwargs, update_cache = false)
        ρρ = updatecache(ρρ)



        tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
        tr_ρ = ITensorNetworks.BeliefPropagationCache(tr_ρ)
        tr_ρ = TN.updatecache(tr_ρ; maxiter = 50, tol = 1e-10)
        println("Trace is $(scalar(tr_ρ))")

        mags = ComplexF64[o for o in trace_expect(tr_ρ, ρ, obs, sphysical, sancilla)]
        println("Total magnetisation is $(sum(mags))")


    end

end

main_hexagonal()