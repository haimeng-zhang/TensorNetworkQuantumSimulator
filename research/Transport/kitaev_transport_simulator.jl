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

using EinExprs

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

function plaquette_weights(ny::Int64, ec, ρρ::AbstractBeliefPropagationCache, sphysical, sancilla)
    return [plaquette_weight(i, ec, ρρ::AbstractBeliefPropagationCache, sphysical, sancilla) for i in 2:(ny - 3)]
end

function plaquette_weight(i::Int64, ec, ρρ::AbstractBeliefPropagationCache, sphysical, sancilla)

    ρρ = copy(ρρ)
    loop = [(i, 1), (i+1, 1), (i+2, 1), (i+2, 2), (i+1, 2), (i, 2)]

    middle_edge = NamedGraphs.PartitionedGraphs.PartitionEdge(NamedEdge((i+1, 1) => (i+1, 2)))

    os, edge_loop = [], NamedEdge[]
    for (j, v) in enumerate(loop)
        e = j <= 5 ? NamedEdge(loop[j]=>loop[j+1]) : NamedEdge(loop[6]=>loop[1])
        push!(edge_loop, e)
        o  =nothing
        #TODO: It's the vertex leaving the plaquette!!!
        vn = neighbors(sphysical, v)
        leaving_vn = nothing
        if first(v) != i + 1
            leaving_vn = only(filter(x -> x ∉ loop, vn))
        else
            if last(v) == 1
                leaving_vn = (first(v), 2)
            else
                leaving_vn = (first(v), 1)
            end
        end
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

    ρρ = TN.sim_edge(ρρ, middle_edge)
    ρOρ = TN.sim_edge(ρOρ, middle_edge)

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

    g = half_periodic_hexagonal_lattice(nx, ny)
    bottom_half_vertices = filter(v -> first(v) <= ny / 2, collect(vertices(g)))

    top_half_vertices = setdiff(collect(vertices(g)), bottom_half_vertices)

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

    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]

    apply_kwargs = (; maxdim = χ, cutoff = 1e-16, normalize = false)


    @show [degree(g, v)  for v in collect(vertices(g))]

    @assert all([degree(g, v) <= 3 for v in collect(vertices(g))])


    ρ = sqrt_high_temperature_initial_state(sphysical, sancilla, 0.0, v -> v ∈ bottom_half_vertices)
    ρρ = build_bp_cache(ρ)
    println("Intial trace is $(scalar(ρρ))")

    A_observables = honeycomb_kitaev_observables(K, ec, bottom_half_vertices)

    B_observables = honeycomb_kitaev_observables(K, ec, top_half_vertices)

    @show length(A_observables)
    @show length(B_observables)

    @assert length(A_observables) == length(B_observables)
    remaining_observables = setdiff(honeycomb_kitaev_observables(K, ec, collect(vertices(g))), [A_observables; B_observables])

    nsteps = 1
    ϵ_top, ϵ_bottom = (mu/nsteps), -(mu/nsteps)
    @show ϵ_top, ϵ_bottom
    layer = honeycomb_kitaev_layer_fourth_order(K, δt, ec)
    layer = layer_to_itensors(layer, sphysical, sancilla)
    init_layer = vcat(honeycomb_kitaev_layer(K, ϵ_bottom, ec, bottom_half_vertices), honeycomb_kitaev_layer(K, ϵ_top, ec, top_half_vertices))
    init_layer = layer_to_itensors(init_layer, sphysical, sancilla; imaginary_time = true)
    for i in 1:nsteps
        ρ, ρρ, errs = apply(init_layer, ρ, ρρ; apply_kwargs)
    end

    Wps = plaquette_weights(ny, ec, ρρ, sphysical, sancilla)

    println("Initial average plaquette flux weight is $(Statistics.mean(abs.(Wps)))")

    ρ, ρρ  = TN.normalize(ρ, ρρ)
    #ρ, ρρ = TN.symmetric_gauge(ρ; cache! = Ref(ρρ))

    A_energies_init = expect(ρρ, sphysical, sancilla, A_observables)
    B_energies_init = expect(ρρ, sphysical, sancilla, B_observables)
    boundary_energies_init = expect(ρρ, sphysical, sancilla, remaining_observables)

    A_energy_init_tot = sum(A_energies_init) + 0.5 * sum(boundary_energies_init)
    B_energy_init_tot = sum(B_energies_init) + 0.5 * sum(boundary_energies_init)

    @show (sum(A_energy_init_tot) + sum(B_energy_init_tot)) / length(vertices(g))
    @show (sum(A_energy_init_tot)) / length(A_observables)
    @show (sum(B_energy_init_tot)) / length(B_observables)

    positions, es_vs_pos = energy_vs_dist(g, A_energies_init, B_energies_init, boundary_energies_init, A_observables, B_observables, remaining_observables)

    # ρρ_BPC = TN.BoundaryMPSCache(copy(ρρ); message_rank = 2*ITensorNetworks.maxlinkdim(ρ), grouping_function = v -> last(v), group_sorting_function = v -> first(v))
    # ρρ_BPC = TN.updatecache(ρρ_BPC; alg = "orthogonal", maxiter = 5, message_update_kwargs = (; niters = 75, tolerance = 1e-12))

    # A_observables_aligned = filter(obs -> last(src(obs[2])) == last(dst(obs[2])),  A_observables)
    # A_energies_aligned_bmps = expect(ρρ_BPC, sphysical, sancilla, A_observables_aligned)
    # A_energies_aligned_bp = expect(ρρ, sphysical, sancilla, A_observables_aligned)
    # @show Statistics.mean(abs.(A_energies_aligned_bmps - A_energies_aligned_bp))
    

    println("Initial energy in A half is $(sum(A_energies_init))")
    println("Initial energy in B half is $(sum(B_energies_init))")
    println("Initial energy in boundary is $(sum(boundary_energies_init))")

    no_trotter_steps = 1000
    measure_freq = 1

    horizontal_A_observable_indices = filter(i -> first(src(A_observables[i][2])) == first(dst(A_observables[i][2])), [i for i in 1:length(A_observables)])

    init_horizontal_energy = sum([A_energies_init[i] for i in horizontal_A_observable_indices])

    t = 0
    f = "/mnt/home/jtindall/ceph/Data/Transport/KitaevModel/"*lattice*"/HeisenbergPictureSqrtApproach/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"K"*string(K)*"nx"*string(nx)
    rows = Int64[r for r in first.(collect(vertices(g)))]
    cols = Int64[r for r in last.(collect(vertices(g)))]
    file_name = f * "TrotterStep0.npz"
    npzwrite(file_name, positions = positions, horizontal_A_energy = init_horizontal_energy, es_vs_pos = es_vs_pos, A_energies = A_energies_init, B_energies = B_energies_init, boundary_energies = boundary_energies_init, rows = rows, cols = cols, in_bottom_half = in_bottom_half)


    for i in 1:no_trotter_steps
        ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs)
        ρ, ρρ = TN.normalize(ρ, ρρ)
        #ρ, ρρ = TN.symmetric_gauge(ρ; cache! = Ref(ρρ))

        if i % measure_freq == 0
            println("Time is $(t)")
            println("Maximum bond dimension is $(ITN.maxlinkdim(ρ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")

            println("Trace is $(scalar(ρρ))")

            Wps = plaquette_weights(ny, ec, ρρ, sphysical, sancilla)
            println("Average plaquette flux weight is $(Statistics.mean(abs.(Wps)))")

            A_energies = expect(ρρ, sphysical, sancilla, A_observables)
            B_energies = expect(ρρ, sphysical, sancilla, B_observables)
            boundary_energies = expect(ρρ, sphysical, sancilla, remaining_observables)

            positions, es_vs_pos = energy_vs_dist(g, A_energies, B_energies, boundary_energies, A_observables, B_observables, remaining_observables)
        
            println("Energy in A half is $(sum(A_energies))")
            println("Energy in B half is $(sum(B_energies))")

            horizontal_A_energy = sum([A_energies[i] for i in horizontal_A_observable_indices])

            A_tot = sum(A_energies) + 0.5 * sum(boundary_energies)

            println("Normalized Energy difference is $((1 / nx)*(A_tot - A_energy_init_tot))")

            println("Horizontal energy transfer is $((1 / nx)*(horizontal_A_energy - init_horizontal_energy))")

            println("Total energy is $(sum(A_energies) + sum(B_energies) + sum(boundary_energies))")

            #ρρ_BPC = TN.BoundaryMPSCache(copy(ρρ); message_rank = 2*ITensorNetworks.maxlinkdim(ρ), grouping_function = v -> last(v), group_sorting_function = v -> first(v))
            #ρρ_BPC = TN.updatecache(ρρ_BPC; alg = "orthogonal", maxiter = 5, message_update_kwargs = (; niters = 75, tolerance = 1e-12))
        
            #A_observables_aligned = filter(obs -> last(src(obs[2])) == last(dst(obs[2])),  A_observables)
            #A_energies_aligned_bmps = expect(ρρ_BPC, sphysical, sancilla, A_observables_aligned)
            #A_energies_aligned_bp = expect(ρρ, sphysical, sancilla, A_observables_aligned)
            #err =  Statistics.mean(abs.(A_energies_aligned_bmps - A_energies_aligned_bp))

            #println("Diff between BP and boundary MPS is $(err)")

            file_name = f * "TrotterStep"*string(i)*".npz"

            npzwrite(file_name, positions = positions, horizontal_A_energy = horizontal_A_energy, Wps = Wps, es_vs_pos = es_vs_pos, A_energies = A_energies, B_energies = B_energies, boundary_energies = boundary_energies, rows = rows, cols = cols, in_bottom_half = in_bottom_half, errs = errs)
        end
        flush(stdout)
        t += δt
        #jldsave("TestWF.jld2", tensornetwork = ρ)
    end
end

function main_heisenberg(lattice::String,χ::Int, ny::Int, mu::Float64, δt::Float64, K::Float64, nx::Int64)

    println("Begining simulation on a $(lattice) lattice with maxdim of $(χ), cylinder length of $(ny), mu of $(mu), dt of $(δt), K of $(K) and nx is $(nx)")

    g = half_periodic_hexagonal_lattice(nx, ny)
    bottom_half_vertices = filter(v -> first(v) <= ny / 2, collect(vertices(g)))

    top_half_vertices = setdiff(collect(vertices(g)), bottom_half_vertices)

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    #ec = edge_color(g, k)

    ec3 = filter(e -> first(src(e)) == first(dst(e)), edges(g))
    ec1, ec2 = NamedEdge[], NamedEdge[]
    for e in setdiff(edges(g), ec3)
        @assert first(src(e)) != first(dst(e))
        r1 = first(src(e)) > first(dst(e)) ? first(dst(e)) : first(src(e))
        if iseven(r1)
            push!(ec1, e)
        else
            push!(ec2, e)
        end
    end

    ec = [ec1, ec2, ec3]

    @assert all([degree(g, v) <= 3 for v in collect(vertices(g))])

    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)

    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]

    apply_kwargs = (; maxdim = χ, cutoff = 1e-12, normalize = false)


    ρ = sqrt_high_temperature_initial_state(sphysical, sancilla, 0.0, v -> v ∈ bottom_half_vertices)
    ρρ = build_bp_cache(ρ)
    println("Intial trace is $(scalar(ρρ))")

    A_observables = honeycomb_kitaev_observables(K, ec, bottom_half_vertices)

    B_observables = honeycomb_kitaev_observables(K, ec, top_half_vertices)

    @show length(A_observables)
    @show length(B_observables)
    remaining_observables = setdiff(honeycomb_kitaev_observables(K, ec, collect(vertices(g))), [A_observables; B_observables])

    nsteps = 1
    ϵ_top, ϵ_bottom = 2*(mu/nsteps), -2*(mu/nsteps)
    layers = honeycomb_kitaev_layers(K, δt, ec)
    layers = [layer_to_itensors(layer, sphysical, sancilla) for layer in layers]
    init_layer = vcat(honeycomb_kitaev_layer(K, ϵ_bottom, ec, bottom_half_vertices), honeycomb_kitaev_layer(K, ϵ_top, ec, top_half_vertices))
    init_layer = layer_to_itensors(init_layer, sphysical, sancilla; imaginary_time = true)
    for i in 1:nsteps
        ρ, ρρ, errs = apply(init_layer, ρ, ρρ; apply_kwargs)
    end

    tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
    tr_ρ = TN.updatecache(ITensorNetworks.BeliefPropagationCache(tr_ρ))
    println("Trace is $(scalar(tr_ρ))")

    A_energies_init = trace_expect(tr_ρ, ρ, sphysical, sancilla, A_observables)

    @show Statistics.mean(A_energies_init)
    B_energies_init = trace_expect(tr_ρ, ρ, sphysical, sancilla, B_observables)
    boundary_energies_init = trace_expect(tr_ρ, ρ, sphysical, sancilla, remaining_observables)

    A_energy_init_tot = sum(A_energies_init) + 0.5 * sum(boundary_energies_init)

    positions, es_vs_pos = energy_vs_dist(g, A_energies_init, B_energies_init, boundary_energies_init, A_observables, B_observables, remaining_observables)

    # ρρ_BPC = TN.BoundaryMPSCache(copy(ρρ); message_rank = 2*ITensorNetworks.maxlinkdim(ρ), grouping_function = v -> last(v), group_sorting_function = v -> first(v))
    # ρρ_BPC = TN.updatecache(ρρ_BPC; alg = "orthogonal", maxiter = 5, message_update_kwargs = (; niters = 75, tolerance = 1e-12))

    # A_observables_aligned = filter(obs -> last(src(obs[2])) == last(dst(obs[2])),  A_observables)
    # A_energies_aligned_bmps = expect(ρρ_BPC, sphysical, sancilla, A_observables_aligned)
    # A_energies_aligned_bp = expect(ρρ, sphysical, sancilla, A_observables_aligned)
    # @show Statistics.mean(abs.(A_energies_aligned_bmps - A_energies_aligned_bp))
    

    println("Initial energy in A half is $(sum(A_energies_init))")
    println("Initial energy in B half is $(sum(B_energies_init))")
    println("Initial energy in boundary is $(sum(boundary_energies_init))")

    no_trotter_steps = 501
    measure_freq = 1

    t = 0
    f = "/mnt/home/jtindall/ceph/Data/Transport/KitaevModel/"*lattice*"/HeisenbergPictureSqrtApproach/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"K"*string(K)*"nx"*string(nx)
    rows = Int64[r for r in first.(collect(vertices(g)))]
    cols = Int64[r for r in last.(collect(vertices(g)))]
    file_name = f * "TrotterStep0.npz"
    npzwrite(file_name, positions = positions, es_vs_pos = es_vs_pos, A_energies = A_energies_init, B_energies = B_energies_init, boundary_energies = boundary_energies_init, rows = rows, cols = cols, in_bottom_half = in_bottom_half)


    for i in 1:no_trotter_steps
        errs = nothing
        for layer in layers
            ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs, update_cache = false)
            ρρ = updatecache(ρρ)
            ρ, ρρ = TN.normalize(ρ, ρρ)
        end

        if i % measure_freq == 0

            tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
            tr_ρ = TN.updatecache(ITensorNetworks.BeliefPropagationCache(tr_ρ))
            println("Trace is $(scalar(tr_ρ))")


            println("Time is $(t)")
            println("Maximum bond dimension is $(ITN.maxlinkdim(ρ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")

            A_energies = trace_expect(tr_ρ, ρ, sphysical, sancilla, A_observables)
            B_energies = trace_expect(tr_ρ, ρ, sphysical, sancilla, B_observables)
            boundary_energies = trace_expect(tr_ρ, ρ, sphysical, sancilla, remaining_observables)

            positions, es_vs_pos = energy_vs_dist(g, A_energies, B_energies, boundary_energies, A_observables, B_observables, remaining_observables)
        
            println("Energy in A half is $(sum(A_energies))")
            println("Energy in B half is $(sum(B_energies))")

            A_tot = sum(A_energies) + 0.5 * sum(boundary_energies)

            println("Normalized Energy difference is $((1 / nx)*(A_tot - A_energy_init_tot))")

            println("Total energy is $(sum(A_energies) + sum(B_energies) + sum(boundary_energies))")

            #ρρ_BPC = TN.BoundaryMPSCache(copy(ρρ); message_rank = 2*ITensorNetworks.maxlinkdim(ρ), grouping_function = v -> last(v), group_sorting_function = v -> first(v))
            #ρρ_BPC = TN.updatecache(ρρ_BPC; alg = "orthogonal", maxiter = 5, message_update_kwargs = (; niters = 75, tolerance = 1e-12))
        
            #A_observables_aligned = filter(obs -> last(src(obs[2])) == last(dst(obs[2])),  A_observables)
            #A_energies_aligned_bmps = expect(ρρ_BPC, sphysical, sancilla, A_observables_aligned)
            #A_energies_aligned_bp = expect(ρρ, sphysical, sancilla, A_observables_aligned)
            #err =  Statistics.mean(abs.(A_energies_aligned_bmps - A_energies_aligned_bp))

            #println("Diff between BP and boundary MPS is $(err)")

            file_name = f * "TrotterStep"*string(i)*".npz"

            npzwrite(file_name, positions = positions, es_vs_pos = es_vs_pos, A_energies = A_energies, B_energies = B_energies, boundary_energies = boundary_energies, rows = rows, cols = cols, in_bottom_half = in_bottom_half, errs = errs)
        end
        flush(stdout)
        t += δt
    end
end

function analyse()
    f = "TestWF.jld2"
    ρ = load(f)["tensornetwork"]

    g = ITensorNetworks.underlying_graph(ρ)
    nx = maximum(last.(vertices(g)))
    ny = maximum(first.(vertices(g)))
    @show ny, nx
    bottom_half_vertices = filter(v -> first(v) <= ny / 2, collect(vertices(g)))
    top_half_vertices = setdiff(collect(vertices(g)), bottom_half_vertices)
    K = 1.0

    ec3 = filter(e -> first(src(e)) == first(dst(e)), edges(g))
    ec1, ec2 = NamedEdge[], NamedEdge[]
    for e in setdiff(edges(g), ec3)
        @assert first(src(e)) != first(dst(e))
        r1 = first(src(e)) > first(dst(e)) ? first(dst(e)) : first(src(e))
        if iseven(r1)
            push!(ec1, e)
        else
            push!(ec2, e)
        end
    end

    ec = [ec1, ec2, ec3]

    @assert all([degree(g, v) <= 3 for v in collect(vertices(g))])

    ρρ = build_bp_cache(ρ)
    println("Trace is $(scalar(ρρ))")

    A_observables = honeycomb_kitaev_observables(K, ec, bottom_half_vertices)

    B_observables = honeycomb_kitaev_observables(K, ec, top_half_vertices)
    remaining_observables = setdiff(honeycomb_kitaev_observables(K, ec, collect(vertices(g))), [A_observables; B_observables])

    sphysical, sancilla = siteinds(ρ), siteinds(ρ)
    for v in vertices(sphysical)
        sphysical[v] = [last(sphysical[v])]
        sancilla[v] = [first(sancilla[v])]
    end
    A_energies = expect(ρρ, sphysical, sancilla, A_observables)
    B_energies = expect(ρρ, sphysical, sancilla, B_observables)
    boundary_energies = expect(ρρ, sphysical, sancilla, remaining_observables)

    @show sum(A_energies) + sum(B_energies) + sum(boundary_energies)

    A_tot = sum(A_energies) + 0.5 * sum(boundary_energies)
    B_tot = sum(B_energies) + 0.5 * sum(boundary_energies)

    println("Total energy in A is  $(2*(1/nx)*(A_tot))")
    println("Total energy in B is  $(2*(1/nx)*(B_tot))")

    # ρρ_BPC = TN.BoundaryMPSCache(copy(ρρ); message_rank = 2, grouping_function = v -> last(v), group_sorting_function = v -> first(v))
    # ρρ_BPC = TN.updatecache(ρρ_BPC; alg = "orthogonal", maxiter = 5, message_update_kwargs = (; niters = 75, tolerance = 1e-12))

    # A_observables_aligned = filter(obs -> last(src(obs[2])) == last(dst(obs[2])),  A_observables)
    # A_energies_aligned_bmps = expect(ρρ_BPC, sphysical, sancilla, A_observables_aligned)
    # A_energies_aligned_bp = expect(ρρ, sphysical, sancilla, A_observables_aligned)
    # err =  maximum(abs.(A_energies_aligned_bmps - A_energies_aligned_bp))

    # println("Diff between BP and boundary MPS is $(err)")

end

#mode, lattice, χ, ny, mu, δt, K, nx = ARGS[1], ARGS[2], parse(Int64, ARGS[3]), parse(Int64, ARGS[4]), parse(Float64, ARGS[5]), parse(Float64, ARGS[6]), parse(Float64, ARGS[7]), parse(Int64, ARGS[8])
mode, lattice, χ, ny, mu, δt, K, nx = "HeisenbergSqrt", "Hexagonal", 16, 20, 0.001, 0.005, 1.0, 2
mode == "HeisenbergSqrt" && main_heisenberg_sqrt(lattice, χ, ny, mu, δt, K, nx)
mode == "Heisenberg" && main_heisenberg(lattice, χ, ny, mu, δt, K, nx)

#analyse()