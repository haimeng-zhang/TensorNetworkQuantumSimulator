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



function main(χ::Int, ny::Int, mu::Float64, δt::Float64, Δ::Float64)

    println("Begining simulation with maxdim of $(χ), cylinder length of $(ny), mu of $(mu), dt of $(δt), Delta of $(Δ)")

    g = named_heavy_hexagonal_cylinder(ny)
    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])
    column_lengths[2] = column_lengths[1]
    column_lengths[4] = column_lengths[3]
    ρ = sqrt_high_temperature_initial_state(sphysical, sancilla, mu, v -> first(v) <= column_lengths[last(v)] / 2)
    ρρ = build_bp_cache(ρ)
    println("Intial trace is $(scalar(ρρ))")

    obs = [("Z", [v]) for v in collect(vertices(g))]

    init_mags = ComplexF64[o for o in expect(ρρ, sphysical, sancilla, obs)]
    println("Initial magnetisation is $(sum(init_mags))")

    top_vertices = filter(v -> first(v) <= column_lengths[last(v)] / 2, collect(vertices(g)))
    init_mag_top = sum([v ∉ top_vertices ? init_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])


    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 3)
    for colored_edges in ec
        _layer = reduce(vcat, [[ITensors.op("RxxyyRzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θxy = 2*δt, θz = 2*Δ*δt), ITensors.op("RxxyyRzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θxy = -2*δt, θz = -2*Δ*δt)] for pair in colored_edges])
        append!(layer, _layer)
    end

    no_trotter_steps = 100
    measure_freq = 1

    t = 0
    f = "/mnt/home/jtindall/ceph/Data/Transport/HeavyHexagonal/HeisenbergPictureSqrtApproach/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"Delta"*string(Δ)

    rows = Int64[r for r in first.(collect(vertices(g)))]
    cols = Int64[r for r in last.(collect(vertices(g)))]
    file_name = f * "TrotterStep0.npz"
    npzwrite(file_name, bp_mags = init_mags, bmps_mags = init_mags, rows = rows, cols = cols)

    apply_kwargs = (; maxdim = χ, cutoff = 1e-10, normalize = false)
    for i in 1:no_trotter_steps
        ρ, ρρ, errs = apply(layer, ρ, ρρ; apply_kwargs)
        ρ, ρρ = TN.normalize(ρ, ρρ)
        ρ, ρρ = TN.symmetric_gauge(ρ; cache! = Ref(ρρ))

        if i % measure_freq == 0
            println("Time is $(t)")
            println("Maximum bond dimension is $(ITN.maxlinkdim(ρ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")

            println("Trace is $(scalar(ρρ))")

            bp_mags = expect(ρρ, sphysical, sancilla, obs)

            file_name = f * "TrotterStep"*string(i)*".npz"
            println("Current BP Measured magnetisation is $(sum(bp_mags))")

            #ρρ_bmps = TN.BoundaryMPSCache(copy(ρρ); message_rank = 3*ITensorNetworks.maxlinkdim(ρ), grouping_function = v -> last(v), group_sorting_function = v -> first(v))
            #ρρ_bmps = TN.updatecache(ρρ_bmps; alg = "orthogonal", maxiter = 6, message_update_kwargs = (; niters = 100, tolerance = 1e-14))
            #bmps_mags = expect(ρρ_bmps, sphysical, sancilla, obs)
            #println("Current BMPS Measured magnetisation is $(sum(bmps_mags))")


            bp_mag_top = sum([v ∉ top_vertices ? bp_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])
            #bmps_mag_top = sum([v ∉ top_vertices ? bmps_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])

            println("Current BP Measured magnetisation transfer is $(bp_mag_top - init_mag_top)")
            #println("Current BMPS Measured magnetisation transfer is $(bmps_mag_top - init_mag_top)")

            bp_mags = ComplexF64[b for b in bp_mags]
            #bmps_mags = ComplexF64[b for b in bmps_mags]
            #npzwrite(file_name, bp_mags = bp_mags, bmps_mags = bmps_mags, rows = rows, cols = cols)
            npzwrite(file_name, bp_mags = bp_mags, rows = rows, cols = cols)
        end
        flush(stdout)
        t += δt
    end
end

#χ, ny, mu, δt, Δ = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4]), parse(Float64, ARGS[5])
χ, ny, mu, δt, Δ =16, 50, 0.1, 0.1, 1.0
main(χ, ny, mu, δt, Δ)
