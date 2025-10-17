using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors: Algorithm, @OpName_str, @SiteType_str, ITensors

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

using JLD2

using Statistics

using Dictionaries: Dictionary, set!

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

function sqrt_high_temperature_initial_state(s::Dictionary, g::NamedGraph, mu, vertex_lower_half_filter)
    return tensornetworkstate(ComplexF64, v -> vertex_lower_half_filter(v) ? (1/sqrt(1+0.25*mu*mu))*[1.0, 0.0, 0.0, -0.5*mu] : (1/sqrt(1+0.25*mu*mu))*[1.0, 0.0, 0.0, +0.5*mu], g, s)
end

function ITensors.op(
    ::OpName"Z", ::SiteType"Pauli"
  )
    mat = zeros(Float64, 4, 4)
    mat[1, 4] = 1
    mat[2, 3] = -1
    mat[3, 2] = 1
    mat[4, 1] = 1
    return mat
end


function main_heisenberg_sqrt(lattice::String, seed::Int, χ::Int, ny::Int, mu::Float64, δt::Float64, J::Float64)

    use_gpu =false
    Random.seed!(12*seed)
    println("Begining simulation on a $(lattice) lattice with maxdim of $(χ), cylinder length of $(ny), mu of $(mu), dt of $(δt), J of $(J)")

    g, bottom_half_vertices = transport_graph_constructor(lattice, ny)

    vs = collect(vertices(g))
    rows = unique(first.(collect(vertices(g))))
    @show nv(g)
    s = siteinds(g, "Pauli")

    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]

    a_vertices = filter(v -> isodd(sum(v)), collect(vertices(g)))
    b_vertices = filter(v -> !isodd(sum(v)), collect(vertices(g)))

    @show length(a_vertices)
    @show length(b_vertices)
    #@assert all([isempty(intersect(neighbors(g, v), a_vertices)) for v in a_vertices])

    ρ = sqrt_high_temperature_initial_state(s, g, mu, v -> v ∈ bottom_half_vertices)

    if use_gpu
        ρ = adapt(CuArray{ComplexF64}, ρ)
    end

    ρ_bpc =BeliefPropagationCache(ρ)
    ρ_bpc = update(ρ_bpc)
    println("Intial trace is $(TN.partitionfunction(ρ_bpc))")

    obs = [("Z", v) for v in collect(vertices(g))]

    init_mags = ComplexF64[o for o in TensorNetworkQuantumSimulator.expect(ρ_bpc, obs)]
    println("Initial mean magnetisation is $(Statistics.mean(init_mags))")

    # mags_vs_row = [Statistics.mean(init_mags[filter(i -> first(vs[i]) == r, [i for i in 1:length(vs)])]) for r in unique(rows)]

    # init_mag_top = sum([v ∉ bottom_half_vertices ? init_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])

    # #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    # k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    # ec = edge_color(g, k)

    # layer = ITensor[]
    # for (i, colored_edges) in enumerate(ec[1:(k)])
    #     θ = J*δt
    #     _layer = reduce(vcat, [[ITensors.op("Rxxyyzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θ = θ),ITensors.op("Rxxyyzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θ = -θ)] for pair in colored_edges])
    #     append!(layer, _layer)
    # end

    # if use_gpu
    #     layer = [adapt(CuArray{ComplexF64}, gate) for gate in layer]
    # end

    # #gate_vertices = length.(TensorNetworkQuantumSimulator.collect_gate_vertices(layer, ρρ))

    # no_trotter_steps = 1000
    # ρ_save_steps = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000,2500]
    # measure_freq = 1

    # t = 0
    # f = "/mnt/home/jtindall/ceph/Data/Transport/"*lattice*"/HeisenbergPictureSqrtApproach/BPMeasurements/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"J"*string(J)

    # rows = Int64[r for r in first.(collect(vertices(g)))]
    # cols = Int64[r for r in last.(collect(vertices(g)))]
    # file_name = f * "TrotterStep0.npz"
    # #npzwrite(file_name, bp_mags = init_mags, bmps_mags = init_mags, rows = rows, cols = cols, in_bottom_half = in_bottom_half, mags_vs_row = mags_vs_row)

    # apply_kwargs = (; maxdim = χ, cutoff = 1e-8, normalize_tensors = false)
    # bp_update_kwargs = (; maxiter=5, tol=1e-4, message_update_alg = Algorithm("posdef_contract"))

    # bp_mags = zeros(ComplexF64, (length(obs), no_trotter_steps + 1))
    # bmps_mags = zeros(ComplexF64, (length(obs), no_trotter_steps + 1))
    # bp_mags[:, 1] = init_mags
    # bmps_mags[:, 1] = init_mags

    # times = [0.0]

    # transferred_mags = ComplexF64[0.0]

    # for i in 1:no_trotter_steps
    #     ρρ, errs = apply_gates(layer, ρρ; apply_kwargs, bp_update_kwargs, inds_per_site = 2)
    #     ρρ = ITensorNetworks.rescale(ρρ; verts = vcat([(v, "ket") for v in vertices(ρ)], [(v, "bra") for v in vertices(ρ)]))

    #     flush(stdout)
    #     t += δt
    #     append!(times, t)

    #     if i % measure_freq == 0
    #         println("Time is $(t)")
    #         println("Maximum bond dimension is $(ITN.maxlinkdim(ρρ))")
    #         println("Average gate fidelity  was $(mean_gate_fidelity(errs))")

    #         println("Trace is $(scalar(ρρ))")

    #         bp_mags[:, i+1] = TN.expect(ρρ, sphysical, sancilla, obs; use_gpu)

    #         mags_vs_row = [Statistics.mean(bp_mags[filter(i -> first(vs[i]) == r, [i for i in 1:length(vs)])]) for r in unique(rows)]

    #         file_name = f * "TrotterStep"*string(i)*".npz"
    #         println("Current BP Measured magnetisation is $(sum(bp_mags))")

    #         bp_mag_top = sum([v ∉ bottom_half_vertices ? bp_mags[j, i+1] : 0 for (j, v) in enumerate(collect(vertices(g)))])

    #         println("Current BP Measured magnetisation transfer is $(bp_mag_top - init_mag_top)")

    #         push!(transferred_mags, bp_mag_top - init_mag_top)

    #         logts, logms = log.(times), log.(transferred_mags)
    #         #alphas = gradient(logms, logts)

    #         i > 1 && println("Rough alpha is $((logms[i] - logms[i-1]) / (logts[i] - logts[i-1]))")

    #         # grouping_function = v -> last(v)
    #         # group_sorting_function  = v -> first(v)
    #         # ρρ_bmps = TensorNetworkQuantumSimulator.BoundaryMPSCache(deepcopy(ρρ); message_rank = χ, group_sorting_function, grouping_function)
    #         # ρρ_bmps = ITensorNetworks.update(ρρ_bmps; alg = "bp", maxiter = 5)

    #         # bmps_mags[:, i+1] = TN.expect(ρρ_bmps, sphysical, sancilla, obs; use_gpu)


    #         # bmps_mag_top = sum([v ∉ bottom_half_vertices ? bmps_mags[j, i+1] : 0 for (j, v) in enumerate(collect(vertices(g)))])

    #         # println("Current BMPS Measured magnetisation is $(sum(bmps_mags))")
    #         # println("Current BMPS Measured magnetisation transfer is $(bmps_mag_top - init_mag_top)")


    #         #npzwrite(file_name, bp_mags = bp_mags, rows = rows,mags_vs_row = mags_vs_row, cols = cols, in_bottom_half = in_bottom_half, errs = errs)
    #     end



    #     if i ∈ ρ_save_steps
    #         file_name = "/mnt/home/jtindall/ceph/Data/Transport/"*lattice*"/HeisenbergPictureSqrtApproach/Results/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"J"*string(J)*"TimeStep"*string(i)*".jld2"
    #         jldsave(file_name, bp_mags = bp_mags, times= times, density_matrix = ρρ, sphysical = sphysical, sancilla = sancilla)
    #     end
        
end

function main_schrodinger(lattice::String, seed::Int, χ::Int, ny::Int, mu::Float64, δt::Float64, J::Float64)

    Random.seed!(12*seed)
    println("Begining simulation on a $(lattice) lattice with maxdim of $(χ), cylinder length of $(ny), mu of $(mu), dt of $(δt), Delta of $(Δ)")

    g, bottom_half_vertices = transport_graph_constructor(lattice, ny)
    @show nv(g)
    s = siteinds("S=1/2", g)
    function up_down(v)
        if v ∈ bottom_half_vertices
            rand() < 0.5*(1 + mu) && return "Z+"
            return "Z-"
        else
            rand() < 0.5*(1 - mu) && return "Z+"
            return "Z-"
        end
    end
    ψ = ITN.ITensorNetwork(v -> up_down(v), s)

    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]

    ψψ = TN.build_bp_cache(ψ)
    obs = [("Z", [v]) for v in collect(vertices(ψ))]

    init_mags = ComplexF64[o for o in TN.expect(ψψ, obs)]
    println("Initial magnetisation is $(sum(init_mags))")

    #Do a k-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    ec = edge_color(g, k)
    _layer = reduce(vcat, [[ITensors.op("Rz", only(s[v]); θ = 2*hz*δt)] for v in vertices(sphysical)])
    append!(layer, _layer)
    for colored_edges in ec
        _layer =[ITensors.op("RxxyyRzz", only(s[src(pair)]), only(s[dst(pair)]); θxy = 2*δt, θz = 2*Δ*δt) for pair in colored_edges]
        append!(layer, _layer)
    end

    no_trotter_steps = 800
    measure_freq = 1

    t = 0
    apply_kwargs = (; maxdim = χ, cutoff = 1e-10)
    f = "/mnt/home/jtindall/ceph/Data/Transport/"*lattice*"/SchrodingerPicture/Seed"*string(seed)*"ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"Delta"*string(Δ)*"hz"*string(hz)

    rows = Int64[r for r in first.(collect(vertices(g)))]
    cols = Int64[r for r in last.(collect(vertices(g)))]
    file_name = f * "TrotterStep0.npz"
    npzwrite(file_name, bp_mags = init_mags, bmps_mags = init_mags, rows = rows, cols = cols, in_bottom_half = in_bottom_half)

    for i in 1:no_trotter_steps
        ψ, ψψ, errs = apply(layer, ψ, ψψ; apply_kwargs)
        ψ, ψψ = normalize(ψ, ψψ)
        ψ, ψψ = TN.symmetric_gauge(ψ; cache! = Ref(ψψ))

        if i % measure_freq == 0
            println("Time is $(t)")
            bp_mags = TN.expect(ψψ, obs)
            println("Maximum bond dimension is $(ITN.maxlinkdim(ψ))")
            println("Average gate fidelity  was $(mean_gate_fidelity(errs))")
            println("Total BP magnetisation is $(sum(bp_mags))")

            #ψψ_bmps = build_boundarymps_cache(ψ, 4*maxlinkdim(ψ); cache_update_kwargs = (; maxiter = 5),
            #cache_construction_kwargs = (; grouping_function = v -> last(v), group_sorting_function = v -> first(v)))
            #bmps_mags = TN.expect(ψψ_bmps, obs)
            bmps_mags = [0 for b in bp_mags]
            println("Total BMPS magnetisation is $(sum(bmps_mags))")

            diffs = sum(abs.(bmps_mags - bp_mags)) / length(bp_mags)
            println("Average diff between bp and mps is $(sum(diffs))")
            bp_mags = ComplexF64[b for b in bp_mags]
            bmps_mags = ComplexF64[b for b in bmps_mags]
            file_name = f * "TrotterStep"*string(i)*".npz"
            npzwrite(file_name, bp_mags = bp_mags, bmps_mags = bmps_mags, rows = rows, cols = cols, in_bottom_half = in_bottom_half, errs = errs)
        end
        t += δt
    end
end

function main_heisenberg(lattice::String, seed::Int, χ::Int, ny::Int, mu::Float64, δt::Float64, J::Float64)

    Random.seed!(12*seed)
    println("Begining simulation on a $(lattice) lattice with maxdim of $(χ), cylinder length of $(ny), mu of $(mu), dt of $(δt), J of $(J)")

    g, bottom_half_vertices = transport_graph_constructor(lattice, ny)
    @show nv(g)
    sphysical = siteinds("S=1/2", g)
    sancilla = siteinds("S=1/2", g)

    ρ = high_temperature_initial_state(sphysical, sancilla, mu, v -> v ∈ bottom_half_vertices)
    ρρ = build_normsqr_bp_cache(ρ)
    println("Intial trace is $(scalar(ρρ))")

    tr_ρ = form_tr_ρ(ρ, sphysical, sancilla)
    tr_ρ = ITensorNetworks.BeliefPropagationCache(tr_ρ)
    tr_ρ = TN.updatecache(tr_ρ; maxiter = 50, tol = 1e-10)
    println("Intial trace is $(scalar(tr_ρ))")

    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]


    obs = [("Z", [v]) for v in collect(vertices(g))]

    init_mags = ComplexF64[o for o in trace_expect(tr_ρ, ρ, obs, sphysical, sancilla)]
    println("Initial magnetisation is $(sum(init_mags))")

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = ITensor[]
    k = lattice ∈ ["Hexagonal", "HeavyHexagonal"] ? 3 : lattice == "Chain" ? 2 : 4
    ec = edge_color(g, k)
    for colored_edges in ec
        _layer = reduce(vcat, [[ITensors.op("Rxxyyzz", only(sphysical[src(pair)]), only(sphysical[dst(pair)]); θ = 2*J*δt), ITensors.op("Rxxyyzz", only(sancilla[src(pair)]), only(sancilla[dst(pair)]); θ = -2*J*δt)] for pair in colored_edges])
        append!(layer, _layer)
    end

    no_trotter_steps = 800
    measure_freq = 1

    t = 0
    f = "/mnt/home/jtindall/ceph/Data/Transport/"*lattice*"/HeisenbergPicture/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"J"*string(J)

    rows = Int64[r for r in first.(collect(vertices(g)))]
    cols = Int64[r for r in last.(collect(vertices(g)))]
    file_name = f * "TrotterStep0.npz"
    #npzwrite(file_name, bp_mags = init_mags, bmps_mags = init_mags, rows = rows, cols = cols)

    init_mag_top = sum([v ∉ bottom_half_vertices ? init_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])

    apply_kwargs = (; maxdim = χ, cutoff = 1e-10, normalize_tensors = true)
    for i in 1:no_trotter_steps
        ρρ, errs = apply_gates(layer, ρρ; apply_kwargs, inds_per_site = 2)

        ρ = ket_network(ρρ)

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
            

            tr_ρ_bmps = TN.BoundaryMPSCache(ITensorNetworks.BeliefPropagationCache(tr_ρ); message_rank = 5*ITensorNetworks.maxlinkdim(ρ),
            grouping_function = v -> last(v), group_sorting_function = v -> first(v))
            tr_ρ_bmps = ITensorNetworks.update(tr_ρ_bmps; alg = "bp", maxiter = 5)
            bmps_mags = trace_expect(tr_ρ_bmps, ρ, obs, sphysical, sancilla)
            println("Current BMPS Measured magnetisation is $(sum(bmps_mags))")

            bp_mags = ComplexF64[b for b in bp_mags]
            bmps_mags = ComplexF64[b for b in bmps_mags]

            bp_mag_top = sum([v ∉ bottom_half_vertices ? bp_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])
            bmps_mag_top = sum([v ∉ bottom_half_vertices ? bmps_mags[i] : 0 for (i, v) in enumerate(collect(vertices(g)))])

            println("Current BP Measured magnetisation transfer is $(bp_mag_top - init_mag_top)")
            println("Current BMPS Measured magnetisation transfer is $(bmps_mag_top - init_mag_top)")
            #npzwrite(file_name, bp_mags = bp_mags, bmps_mags = bmps_mags, rows = rows, cols = cols, in_bottom_half = in_bottom_half, errs = errs)
        end
        flush(stdout)
        t += δt
    end
end

#mode, lattice, χ, ny, mu, δt, J, seed = ARGS[1], ARGS[2], parse(Int64, ARGS[3]), parse(Int64, ARGS[4]), parse(Float64, ARGS[5]), parse(Float64, ARGS[6]), parse(Float64, ARGS[7]), parse(Int64, ARGS[8])
#mode, lattice, χ, ny, mu, δt, J, seed = "HeisenbergSqrt", "Hexagonal", 16, 50, 0.05, 0.05, 1.0, 1
mode, lattice, χ, ny, mu, δt, J, seed = "HeisenbergSqrt", "Square", 8, 40, 0.05, 0.05, 1.0, 1
mode == "HeisenbergSqrt" && main_heisenberg_sqrt(lattice, seed, χ, ny, mu, δt, J)
#mode == "Heisenberg" && main_heisenberg(lattice, seed, χ, ny, mu, δt, J)
#mode == "Schrodinger" && main_schrodinger(lattice, seed, χ, ny, mu, δt, J)
