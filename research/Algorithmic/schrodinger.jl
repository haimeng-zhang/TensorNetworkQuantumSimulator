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

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str

using Random

using EinExprs: Greedy

using NPZ

using Base.Threads
using MKL
using LinearAlgebra

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
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

#Method 1 for measuring observale
function measure_high_weight_obs_V1(ψψ, obs)
    ψOψ = TN.insert_observable(ψψ, obs)
    verts = [(v, "operator") for v in last(obs)]
    partitions = NamedGraphs.PartitionedGraphs.partitionvertices(ψψ, verts)
    ms = ITensorNetworks.incoming_messages(ψψ, partitions)
    local_numer_tensors = ITensorNetworks.factors(ψOψ, partitions)
    local_denom_tensors = ITensorNetworks.factors(ψψ, partitions)
    ts = [ms; local_numer_tensors]
    seq = ITensorNetworks.contraction_sequence(ts; alg = "einexpr", optimizer = Greedy())
    numer = ITensors.contract(ts; sequence=seq)[]

    ts = [ms; local_denom_tensors]
    seq = ITensorNetworks.contraction_sequence(ts; alg = "einexpr", optimizer = Greedy())
    denom =  ITensors.contract(ts; sequence=seq)[]
    return numer / denom
end

#Method 2 for measuring observale
function measure_high_weight_obs_V2(ψψ, obs)
    ψOψ = TN.insert_observable(ψψ, obs)
    ψOψ = updatecache(ψOψ)
    return scalar(ψOψ; alg = "bp") / scalar(ψψ; alg = "bp")
end

function main(seed::Int, maxdim::Int, S::Int, grad::Float64, delta::Float64)
    #Input file location
    root = "/mnt/home/jtindall/ceph/Data/Algorthmic/Circuits/"
    f = root *"qa_circuit_W=4_S="*string(S)*"_J=0.39269908169872414_b=0.39269908169872414_h0=0.39269908169872414_grad="*string(grad)*"_delta="*string(delta)*".txt"
    circuit = read_qasm_circuit(f)
    g = graph_from_circuit(circuit)

    #Look at Gates involved
    @show unique(first.(circuit))
    #Number of gates
    @show length(circuit)

    #Get the physical indices
    s = ITN.siteinds("S=1/2", g)
    #Sites to initial "Z" on
    z_vertices = [45, 47, 67, 69, 89, 91]


    #Random initial bitstring, use a seed
    Random.seed!(seed)
    initial_bitstring = Dictionary(vertices(g), [rand([-1, 1]) for v in vertices(g)])

    #Initial state based off bitstring
    ψ = ITensorNetwork(v -> initial_bitstring[v] == -1 ? "Z-" : "Z+", s)

    #Coeff that this term contributed to the signal (signal is sum over random states multiplied by this coeff)
    init_coeff = prod([initial_bitstring[v] for v in z_vertices])

    #Build the bp cache
    ψψ = build_bp_cache(ψ)

    #Measure things
    st = NamedGraphs.steiner_tree(g, z_vertices)
    obs = (prod([v ∈ z_vertices ? "Z" : "I" for v in vertices(st)]), collect(vertices(st)))
    O_init_V1, O_init_V2 = measure_high_weight_obs_V1(ψψ, obs), measure_high_weight_obs_V2(ψψ, obs)

    println("Initial value of O is $O_init_V1 (first method) or $O_init_V2 (second method)")

    #It should 
    @assert isapprox(abs(O_init_V1 - O_init_V2), 0; atol =1e-8)
    @assert isapprox(abs(O_init_V1 - init_coeff), 0; atol = 1e-8)

    #Transform circuit to itensors
    U = TN.toitensor(circuit, s)

    #Apply kwargs (maximum bond dimension)
    apply_kwargs = (; maxdim = maxdim, cutoff = 1e-10, normalize = true)

    #Apply the circuit to get O. U
    t = time()
    ψ, ψψ, errs = apply(U, ψ, ψψ; apply_kwargs, verbose = true)
    t = time() - t
    println("Simulation O -> OU took $(t) seconds")
    println("Average gate error was $(Statistics.mean(errs))")
    println("Rough fedility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

    O_final_V1, O_final_V2 = measure_high_weight_obs_V1(ψψ, obs), measure_high_weight_obs_V2(ψψ, obs)

    println("Final value of O is $O_final_V1 (first method) or $O_final_V2 (second method)")

    save_file = "/mnt/home/jtindall/ceph/Data/Algorthmic/Schrodinger/Seed"*string(seed)*"Maxdim"*string(maxdim)*"S"*string(S)*"grad"*string(grad)*"delta"*string(delta)*".npz"
    npzwrite(save_file, O_init = O_init_V1, O_final_V1 = O_final_V1, O_final_V2 = O_final_V2, errs = errs)
end

seed, χ, S, grad, delta = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3]), parse(Float64, ARGS[4]), parse(Float64, ARGS[5])
#seed, χ, S, grad, delta = 1, 256, 3, 0.05, 0.011
main(seed, χ, S, grad, delta)