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

using Statistics

using JLD2

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
    os = ComplexF64[]
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

function trace_expect(tr_ρ::AbstractBeliefPropagationCache, ρ::ITensorNetwork, sphysical, sancilla, obs::Vector{<:Tuple})
    os = ComplexF64[]
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

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obss::Vector)
    return [expect(ρρ, sphysical, sancilla, obs) for obs in obss]
end

function layer_to_itensors(layer, sphysical, sancilla; imaginary_time = false)
    ts = []
    for gate in layer
        θ = gate[3]
        ϕp = imaginary_time ? im * 0.25 * θ : 0.5 * θ
        ϕa = imaginary_time ? im * 0.25 * θ : -0.5 * θ
        #Rotations from utils.jl are doubled, hence the halving here
        t1 = ITensors.op(gate[1], only(sphysical[src(gate[2])]), only(sphysical[dst(gate[2])]); ϕ = 0.5*ϕp)
        t2 = ITensors.op(gate[1], only(sancilla[src(gate[2])]), only(sancilla[dst(gate[2])]); ϕ = 0.5*ϕa)
        push!(ts, t1)
        push!(ts, t2)
    end
    return ts
end

function energy_vs_dist(g, A_energies, B_energies, boundary_energies, A_observables, B_observables, boundary_observables)
    positions = unique(first.(collect(vertices(g))))
    es = [A_energies; B_energies; boundary_energies]
    obs = [A_observables; B_observables; boundary_observables]

    es_obs = [(e, o) for (e,o) in zip(es, obs)]
    es = ComplexF64[]

    for pos in positions
        relevant_obs = filter(o -> first(src(o[2][2])) == pos || first(dst(o[2][2])) == pos, es_obs)
        e = sum(first.(relevant_obs))
        push!(es, e)
    end

    return positions, es
end

function plaquette_weights(ec, ρρ::AbstractBeliefPropagationCache, sphysical, sancilla)
    egs = NamedGraphs.edgeinduced_subgraphs_no_leaves(NamedGraphs.PartitionedGraphs.partitioned_graph(ρρ), 6)
    return [plaquette_weight(eg, ec, ρρ::AbstractBeliefPropagationCache, sphysical, sancilla) for eg in egs]
end

function plaquette_weight(eg, ec, ρρ::AbstractBeliefPropagationCache, sphysical, sancilla)

    ρρ = copy(ρρ)
    loop = collect(vertices(eg))

    os, edge_loop = [], edges(eg)
    @assert length(edge_loop) == 6
    for (j, v) in enumerate(loop)
        o  =nothing
        #TODO: It's the vertex leaving the plaquette!!!
        vn = neighbors(sphysical, v)
        leaving_vn = only(filter(x -> x ∉ loop, vn))
        _e = NamedEdge(v => leaving_vn)
        if _e ∈ ec[1] || reverse(_e) ∈ ec[1]
            o = "X"
        elseif _e ∈ ec[2] || reverse(_e) ∈ ec[2]
            o = "Y"
        elseif _e ∈ ec[3] || reverse(_e) ∈ ec[3]
            o = "Z"
        end
        push!(os, o)
    end

    ρOρ = copy(ρρ)
    for (j, v) in enumerate(loop)
        ITensorNetworks.@preserve_graph ρOρ[(v,"operator")] = ITensors.op("Id", only(sancilla[v])) * ITensors.op(os[j], only(sphysical[v]))
    end

    incoming_pes = ITensorNetworks.boundary_partitionedges(ρρ,NamedGraphs.PartitionedGraphs.PartitionEdge.(edge_loop))
    @assert length(incoming_pes) == 6

    numer_factors = reduce(vcat, [ITensorNetworks.factors(ρOρ, NamedGraphs.PartitionedGraphs.PartitionVertex(v)) for v in loop])
    denom_factors = reduce(vcat, [ITensorNetworks.factors(ρρ, NamedGraphs.PartitionedGraphs.PartitionVertex(v)) for v in loop])


    incoming_numer_messages = [only(ITensorNetworks.message(ρOρ, pe)) for pe in incoming_pes]
    incoming_denom_messages = [only(ITensorNetworks.message(ρρ, pe)) for pe in incoming_pes]

    numer_ts = [numer_factors; incoming_numer_messages]
    numer = ITensors.contract(numer_ts; sequence = ITensorNetworks.contraction_sequence(numer_ts; alg="einexpr", optimizer=Greedy()))

    denom_ts = [denom_factors; incoming_denom_messages]
    denom = ITensors.contract(denom_ts; sequence = ITensorNetworks.contraction_sequence(denom_ts; alg="einexpr", optimizer=Greedy()))

    return numer[] / denom[]
end


function main_heisenberg_sqrt(lattice::String,χ::Int, ny::Int, mu::Float64, δt::Float64, K::Float64, nx::Int64)

    println("Begining simulation on a $(lattice) lattice with maxdim of $(χ), cylinder length of $(ny), mu of $(mu), dt of $(δt), K of $(K) and nx is $(nx)")

    g = NamedGraphs.NamedGraphGenerators.named_hexagonal_lattice_graph(4,4; periodic = true)
    bottom_half_vertices = filter(v -> first(v) <= ny / 2, collect(vertices(g)))

    top_half_vertices = setdiff(collect(vertices(g)), bottom_half_vertices)

    egs = NamedGraphs.edgeinduced_subgraphs_no_leaves(g, 6)
    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    #ec = edge_color(g, k)

    ec3 = filter(e -> first(src(e)) == first(dst(e)), edges(g))
    ec1, ec2 = NamedEdge[], NamedEdge[]
    for e in setdiff(edges(g), ec3)
        @assert first(src(e)) != first(dst(e))
        @assert last(src(e)) == last(dst(e))
        r1 = first(src(e)) > first(dst(e)) ? first(dst(e)) : first(src(e))
        col = last(src(e))

        group_1_edge = (iseven(r1) && isodd(col)) || (isodd(r1) && iseven(col))
        if group_1_edge
            push!(ec1, e)
        else
            push!(ec2, e)
        end
    end

    ec = [ec1, ec2, ec3]

    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)

    apply_kwargs = (; maxdim = χ, cutoff = 1e-16, normalize = false)


    @show [degree(g, v)  for v in collect(vertices(g))]

    @assert all([degree(g, v) <= 3 for v in collect(vertices(g))])


    ρ = sqrt_high_temperature_initial_state(sphysical, sancilla, 0.0, v -> v ∈ bottom_half_vertices)
    ρρ = build_bp_cache(ρ)
    println("Intial trace is $(scalar(ρρ))")

    observables = honeycomb_kitaev_observables(K, ec, collect(vertices(g)))

    dβ = 0.1
    nsteps = 100
    layer = honeycomb_kitaev_layer(K, δt, ec)
    layer = layer_to_itensors(layer, sphysical, sancilla)
    init_layer = layer_to_itensors(honeycomb_kitaev_layer(K, dβ, ec), sphysical, sancilla; imaginary_time = true)
    for i in 1:nsteps
        println("On Trotter step $(i)")
        ρ, ρρ, errs = apply(init_layer, ρ, ρρ; apply_kwargs)
        ρ, ρρ  = TN.normalize(ρ, ρρ)

        Wp = plaquette_weight(first(egs), ec, ρρ, sphysical, sancilla)

        println("Average plaquette flux weight is $(abs(Wp))")
    end


    energies = expect(ρρ, sphysical, sancilla, observables)

    e_gs = (sum(energies)) / length(vertices(g))
    

    println("Energy is $(e_gs)")
end

   
mode, lattice, χ, ny, mu, δt, K, nx = "HeisenbergSqrt", "Hexagonal", 12, 30, 0.001, 0.0025, 1.0, 2
mode == "HeisenbergSqrt" && main_heisenberg_sqrt(lattice, χ, ny, mu, δt, K, nx)
