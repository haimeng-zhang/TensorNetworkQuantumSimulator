using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using ITensorNetworks: BeliefPropagationCache, IndsNetwork, AbstractBeliefPropagationCache
using ITensors: Index, ITensor, inner, itensor, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, Algorithm, datatype
using NamedGraphs.PartitionedGraphs: PartitionEdge

using Random

using EinExprs: Greedy

using NPZ

using Base.Threads
using MKL
using LinearAlgebra

using CUDA

using Adapt: adapt

BLAS.set_num_threads(min(2, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

#Some gate definitions needed
function ITensors.op(
    ::OpName"S", ::SiteType"S=1/2"
  )
    mat = zeros(ComplexF64, 2, 2)
    mat[1, 1] = 1.0
    mat[2, 2] = 1.0 * im
    return mat
end

function ITensors.op(
    ::OpName"Sx", ::SiteType"S=1/2"
  )
    mat = zeros(ComplexF64, 2, 2)
    mat[1, 1] = 0.5*(1.0 + 1.0*im)
    mat[2, 2] = 0.5*(1.0 + 1.0*im)
    mat[1, 2] = 0.5*(1.0 - 1.0*im)
    mat[2, 1] = 0.5*(1.0 - 1.0*im)
    return mat
end

function ITensors.op(
    ::OpName"Sdg", ::SiteType"S=1/2"
  )
  mat = zeros(ComplexF64, 2, 2)
  mat[1, 1] = 1.0
  mat[2, 2] = -1.0 * im
  return mat
end

function ITensors.op(
    ::OpName"Sxdg", ::SiteType"S=1/2"
  )
  mat = zeros(ComplexF64, 2, 2)
  mat[1, 1] = 0.5*(1.0 - 1.0*im)
  mat[2, 2] = 0.5*(1.0 - 1.0*im)
  mat[1, 2] = 0.5*(1.0 + 1.0*im)
  mat[2, 1] = 0.5*(1.0 + 1.0*im)
  return mat
end

#Parse a QASM gate into a readable format for this repo (gate_str, qubits_acted_on, possible_kwarg)
function parse_gate(s::String)
    qs = [parse(Int64, q.match[2:(length(q.match) - 1)]) for q in eachmatch(r"\[(.*?)\]", s)]
    kwarg = match(r"\((.*?)\)", s)
    if !isnothing(kwarg)
        kwarg = eval(Meta.parse(kwarg.match[2:(length(kwarg.match) - 1)]))
    end

    gate_str = isnothing(kwarg) ?  match(r"^[^\ ]+", s).match : match(r"^[^\(]+", s).match
    gate_str = uppercasefirst(gate_str)

    isnothing(kwarg) && return (gate_str, qs)
    return (gate_str, qs, kwarg)
end

#Read a full circuit from the given text file
function read_qasm_circuit(f::String)
    file = open(f)
    lines = readlines(file)
    close(file)
    n_lines = length(lines)
    gates = []
    for i in 4:n_lines
        line = lines[i]
        if !contains(line, "barrier")
            push!(gates, parse_gate(line))
        end
    end
    return gates
end

#Given a circuit of two-site and 1-site gates, build the graph induced by the circuit
function graph_from_circuit(circ)
    g = NamedGraph()
    for gate in circ
        qubits = gate[2]

        if length(qubits) == 2
            v1, v2 = first(qubits), last(qubits)
            !has_vertex(g, v1) && add_vertex!(g, v1)
            !has_vertex(g, v2) && add_vertex!(g, v2)
            !has_edge(g, NamedEdge(v1 => v2)) && add_edge!(g, NamedEdge(v1 => v2))
        end
    end
    return g
end

function measure_observable_V1(ψ_bpc::BeliefPropagationCache, s::IndsNetwork, obs)
    dtype = ITensors.datatype(ψ_bpc)
    ops, verts, coeff = TN.collectobservable(obs)
    ket_tensors = [ψ_bpc[v] for v in verts]
    bra_tensors = dag.(prime.(ket_tensors))
    numer_operators = [adapt(dtype, ITensors.op(op_str, only(s[v]))) for (op_str, v) in zip(ops, verts)]
    denom_operators = [adapt(dtype, ITensors.op("Id", only(s[v]))) for  v in verts]
    ms = ITensorNetworks.incoming_messages(ψ_bpc, ITensorNetworks.partitionvertices(ψ_bpc, verts))

    numer_ts = [ket_tensors; bra_tensors; numer_operators; ms]
    seq = ITensorNetworks.contraction_sequence(numer_ts; alg = "einexpr", optimizer = Greedy())
    numer = ITensors.contract(numer_ts; sequence = seq)[]

    denom_ts = [ket_tensors; bra_tensors; denom_operators; ms]
    seq = ITensorNetworks.contraction_sequence(denom_ts; alg = "einexpr", optimizer = Greedy())
    denom = ITensors.contract(denom_ts; sequence = seq)[]

    return coeff * (numer/denom)
end

function special_region_scalar(bpc::BeliefPropagationCache, v, ops, verts)
    incoming_ms = ITensorNetworks.incoming_messages(bpc, v)
    state = only(ITensorNetworks.factors(bpc, v))
    state_dag = dag(prime(state))

    if parent(v) ∉ verts
        op = ITensors.op("I", only(inds(state; tags = "Site")))
    else
        op_index = findfirst(_v -> _v ==  parent(v), verts)
        op = ITensors.op(ops[op_index], only(inds(state; tags = "Site")))
    end
    op = adapt(datatype(bpc))(op)

    contract_list = ITensor[incoming_ms; [state, op, state_dag]]
    sequence = ITensorNetworks.contraction_sequence(contract_list; alg = "optimal")
    return ITensors.contract(contract_list; sequence = sequence)[]
end

function special_message_update!(bpc::BeliefPropagationCache, e, ops, verts)
    incoming_ms = ITensorNetworks.incoming_messages(bpc, src(e);  ignore_edges = PartitionEdge[reverse(e)])
    state = only(ITensorNetworks.factors(bpc, src(e)))
    state_dag = dag(prime(state))
    if parent(src(e)) ∉ verts
        op = ITensors.op("I", only(inds(state; tags = "Site")))
    else
        op_index = findfirst(_v -> _v == parent(src(e)), verts)
        op = ITensors.op(ops[op_index], only(inds(state; tags = "Site")))
    end
    op = adapt(datatype(bpc))(op)

    contract_list = ITensor[incoming_ms; [state, op, state_dag]]
    sequence = ITensorNetworks.contraction_sequence(contract_list; alg = "optimal")
    m = ITensors.contract(contract_list; sequence)
    m /= norm(m)
    return ITensorNetworks.set_message!(bpc, e, ITensor[m])
end

function measure_observable_V2(ψ_bpc::BeliefPropagationCache, obs; niters = 10)
    ops, verts, coeff = TN.collectobservable(obs)

    ψ_bpc = ITensorNetworks.rescale_messages(ψ_bpc)
    log_denom = sum(log.([special_region_scalar(ψ_bpc, pv, [], []) for pv in ITensorNetworks.partitionvertices(ψ_bpc)]))
    for i in 1:niters
        for pe in ITensorNetworks.default_bp_edge_sequence(ψ_bpc)
            special_message_update!(ψ_bpc, pe, ops, verts)
        end
    end

    ψ_bpc = ITensorNetworks.rescale_messages(ψ_bpc)
    log_numer = sum(log.([special_region_scalar(ψ_bpc, pv, ops, verts) for pv in ITensorNetworks.partitionvertices(ψ_bpc)]))

    return coeff * exp(log_numer - log_denom)
end

function insert_projector(ψIψ::AbstractBeliefPropagationCache, verts, config)

    op_strings = [c == '0' ? "Proj0" : "Proj1" for c in config]
    ψIψ_vs = [ψIψ[(v, "operator")] for v in verts]
    sinds =
        [commonind(ψIψ[(v, "ket")], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    operators = [adapt(datatype(ψIψ[(v, "operator")]))(ITensors.op(op_strings[i], sinds[i])) for (i, v) in enumerate(verts)]

    ψOψ = ITensorNetworks.update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))
    return ψOψ
end

function ITensorNetworks.set_default_kwargs(alg::Algorithm"adapt_square_update")
    return Algorithm("adapt_square_update"; adapt=alg.kwargs.adapt, normalize=true, sequence_alg = "optimal")
end

function ITensorNetworks.updated_message(
    alg::Algorithm"adapt_square_update", bpc::AbstractBeliefPropagationCache, edge::PartitionEdge
  )

    state = only(ITensorNetworks.factors(bpc, src(edge)))
    adapted_state = CUDA.cu(state)
    adapted_factors = [adapted_state, noprime(dag(prime(adapted_state)), tags = "Site")]
    adapted_messages = [CUDA.cu(m) for m in ITensorNetworks.incoming_messages(bpc, src(edge); ignore_edges = PartitionEdge[reverse(edge)])]

    contract_list = ITensor[adapted_messages; adapted_factors]
    sequence = ITensorNetworks.contraction_sequence(contract_list; alg = alg.kwargs.sequence_alg)
    m = make_hermitian(ITensors.contract(contract_list; sequence))
    message_norm = norm(m)
    if alg.kwargs.normalize && !iszero(message_norm)
        m /= message_norm
    end
    updated_messages = ITensor[m]
    dtype = mapreduce(datatype, promote_type, ITensorNetworks.message(bpc, edge))
    return map(adapt(Vector{ComplexF32}), updated_messages)
  end

function main(seed::Int, maxdim::Int, L::Int64, b::Float64, delta::Float64, _version::Int64)
    version = _version == 1 ? "" : "V2"
    #Input file location
    root = "/mnt/home/jtindall/ceph/Data/Algorthmic/Circuits/49q_circuits/"
    f = root *"49q_FL="*string(L)*"_b="*string(b)*"_delta="*string(delta)*version*".txt"
    circuit = read_qasm_circuit(f)
    g = graph_from_circuit(circuit)

    #Look at Gates involved
    @show unique(first.(circuit))
    #Number of gates
    @show length(circuit)

    #Get the physical indices
    s = ITN.siteinds("S=1/2", g)
    #Sites to initial "Z" on
    z_vertices = [52,59,72]


    #Random initial bitstring, use a seed
    Random.seed!(seed)
    initial_bitstring = Dictionary(vertices(g), [rand([-1, 1]) for v in vertices(g)])

    #Initial state based off bitstring
    ψ = ITensorNetwork(ComplexF32, v -> initial_bitstring[v] == -1 ? "Z-" : "Z+", s)

    #If you want BP to run on GPU instead of CPU
    #bp_update_kwargs = (; maxiter = 25, tol = 1e-5, message_update_alg = Algorithm("adapt_square_update"; adapt = CUDA.cu))

    #If you want BP to run on CPU
    bp_update_kwargs = (; maxiter = 25, tol = 1e-5, message_update_alg = Algorithm("squarebp"))
    
    ψ_bpc = ITensorNetworks.BeliefPropagationCache(ψ)
    TN.initialize_square_bp_messages!(ψ_bpc)
    ψ_bpc = ITensorNetworks.update(ψ_bpc; bp_update_kwargs...)

    #If you want everything to run on GPU
    ψ_bpc = CUDA.cu(ψ_bpc)

    #Measure things
    st = NamedGraphs.steiner_tree(g, z_vertices)
    obs = (prod([v ∈ z_vertices ? "Z" : "I" for v in vertices(st)]), collect(vertices(st)))
    O_init = measure_observable_V2(ψ_bpc, obs)

    println("Initial value of O is $O_init")

    #Apply kwargs (maximum bond dimension)
    apply_kwargs = (; maxdim = maxdim, cutoff = 1e-10, normalize_tensors = true)

    #Apply the circuit to get O. U
    t = time()
    ψ_bpc, errs = apply_gates(circuit, ψ_bpc; bp_update_kwargs, apply_kwargs, verbose = true, transfer_to_gpu = false)
    t = time() - t
    println("Simulation O -> OU took $(t) seconds")
    println("Average gate error was $(Statistics.mean(errs))")
    println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

    #O_final_V1 = measure_observable_V1(ψ_bpc, s, obs)
    O_final_V2 = measure_observable_V2(ψ_bpc, obs)

    label = _version == 1 ? "CZCircuitb0.25" : "FractionalRzzCircuit"
    save_file = "/mnt/home/jtindall/ceph/Data/Algorthmic/Schrodinger/Seed"*string(seed)*"Maxdim"*string(maxdim)*"FL"*string(L)*"delta"*string(delta)*label*".npz"
    npzwrite(save_file, O_init = O_init, O_final = O_final_V2, errs = errs)
end


function run_over_stuff()
    seeds = [1,2,3,4,5]
    χs = [196]
    deltas = [0.0, 0.05, 0.1, 0.15,0.2]
    L = 6
    b= 0.25
    _versions = [1]

    for _version in _versions
        for χ in χs
            println("Chi is $(χ)")
            for delta in deltas
                println("Delta is $(delta)")
                for seed in seeds
                    println("Seed is $(seed)")
                    main(seed, χ, L, b, delta, _version)
                    CUDA.reclaim()
                    GC.gc()
                end
            end
        end
    end
end

run_over_stuff()