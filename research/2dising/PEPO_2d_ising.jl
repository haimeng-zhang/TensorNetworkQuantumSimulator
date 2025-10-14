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

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using CUDA
using Adapt
using Dictionaries

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))


function identity_state(sphysical, sancilla; normalize = false)
    ψ = ITensorNetworks.random_tensornetwork(Float32, sphysical; link_space = 1)
    for v in vertices(ψ)
        array = Float32[1 0; 0 1]
        λ = normalize ? 0.5 : 1 
        ITensorNetworks.@preserve_graph ψ[v] = λ * ITensors.ITensor(Float32, array, only(sphysical[v]), only(sancilla[v]))
    end
    ψ = ITensorNetworks.insert_linkinds(ψ)
    return ψ
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
        ITensorNetworks.@preserve_graph ρOρ[(v,"operator")] = adapt(Vector{Float32})(ITensors.op("Id", only(sancilla[v])) * ITensors.op(op_vec[i], only(sphysical[v])))
    end

    numerator = ITensorNetworks.region_scalar(ρOρ, [(v, "ket") for v in vs])
    denominator = ITensorNetworks.region_scalar(ρρ, [(v, "ket") for v in vs])

    return coeff * numerator / denominator
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obss::Vector{<:Tuple})
    return [expect(ρρ, sphysical, sancilla, obs) for obs in obss]
end

function combine_ρ(ρ::ITensorNetwork, combiners)
    ρ = copy(ρ)
    for v in vertices(ρ)
        ρ[v] *= combiners[v]
    end

    return ρ
end

function main()

    n = 6
    grid = named_grid((n,n))
    sphysical = siteinds("S=1/2", grid)
    sancilla = siteinds("S=1/2", grid)

    combiners = Dictionary(collect(vertices(grid)), [adapt(Vector{Float32})(ITensors.combiner(only(sphysical[v]), only(sancilla[v]))) for v in vertices(grid)])

    ρ = identity_state(sphysical, sancilla)
    ρ = combine_ρ(ρ, combiners)
    δβ = 0.01 
    g = -3.1
    J = -1

    #Do a custom 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    ec1 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 1:2:(n-1)] for i in 1:n])
    ec2 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 2:2:(n-1)] for i in 1:n])
    ec3 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 1:2:(n-1)] for i in 1:n])
    ec4 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 2:2:(n-1)] for i in 1:n])
    ec = [ec1, ec2, ec3, ec4]

    @assert length(reduce(vcat, ec)) == length(edges(grid))
    nsteps = 50
    apply_kwargs = (; maxdim = 4, cutoff = 1e-12)

    MPS_message_rank = 10
    
    β = 0
    for i in 1:nsteps
        println("Inverse Temperature is $β")
        println("Bond dimension of PEPO $(ITensorNetworks.maxlinkdim(ρ))")

        #Apply the singsite rotations half way
        for v in vertices(grid)
            g1, g2 = TN.toitensor(("Rx", [v], -0.25 * im * g *δβ), sphysical), TN.toitensor(("Rx", [v], - 0.25 *im * g *δβ), sancilla)
            gate = g1 * g2 * combiners[v] * dag(prime(combiners[v]))
            gate = adapt(Vector{Float32})(gate)
            ρ[v] = normalize(apply(gate, ρ[v]))
        end

        #Apply the two site rotations, use a boundary MPS cache to apply them (need to run column or row wise depending on the gates)
        for (k, colored_edges) in enumerate(ec)
            if k == 1 || k == 2
                grouping_function = v -> last(v)
                group_sorting_function = v -> first(v)
            else
                grouping_function = v -> first(v)
                group_sorting_function = v -> last(v)
            end
            
            ρρ = TN.build_normsqr_bmps_cache(ρ, MPS_message_rank; cache_construction_kwargs = (; grouping_function, group_sorting_function))
            for pair in colored_edges
                g1, g2 = TN.toitensor(("Rzz", pair, -im * 0.5* J *δβ), sphysical), TN.toitensor(("Rzz", pair, -im * J * 0.5* δβ), sancilla)
                gate = g1 * g2 * combiners[src(pair)] * combiners[dst(pair)] * dag(prime(combiners[src(pair)])) * dag(prime(combiners[dst(pair)]))
                gate = adapt(Vector{Float32})(gate)
                envs = ITensorNetworks.environment(ρρ, [(src(pair), "ket"), (dst(pair), "ket"), (src(pair), "bra"), (dst(pair), "bra"), (src(pair), "operator"), (dst(pair), "operator")])
                ρv1, ρv2  = ITensorNetworks.full_update_bp(gate, ρ, [src(pair), dst(pair)]; envs, apply_kwargs...)
                ρ[src(pair)], ρ[dst(pair)] = normalize(ρv1), normalize(ρv2)
            end
        end


        for v in vertices(grid)
            g1, g2 = TN.toitensor(("Rx", [v], -0.25 * im * g *δβ), sphysical), TN.toitensor(("Rx", [v], -0.25 * im * g *δβ), sancilla)
            gate = g1 * g2 * combiners[v] * dag(prime(combiners[v]))
            gate = adapt(Vector{Float32})(gate)
            ρ[v] = normalize(apply(gate, ρ[v]))
        end

        β += δβ

        ρ_uncombined = combine_ρ(ρ, combiners)
        tr_ρ = form_tr_ρ(ρ_uncombined, sphysical, sancilla)
        tr_ρ_bmps = TensorNetworkQuantumSimulator.BoundaryMPSCache(tr_ρ; message_rank = MPS_message_rank)
        tr_ρ_bmps = ITensorNetworks.update(tr_ρ_bmps)
        ρρ_uncombined = TN.build_normsqr_bmps_cache(ρ_uncombined, MPS_message_rank)
        sz_doubled = TN.expect(ρρ_uncombined, sphysical, sancilla, ("X", [(2,2)]))
        sz_single = only(trace_expect(tr_ρ_bmps, ρ_uncombined, [("X", [(2,2)])], sphysical, sancilla))

        println("Expectation value at beta  = $(β) is $(sz_single)")
        println("Expectation value at beta  = $(2*β) is $(sz_doubled)")
    end


end

main()