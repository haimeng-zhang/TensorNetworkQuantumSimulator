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

using NPZ

using Base.Threads
using MKL
using LinearAlgebra

BLAS.set_num_threads(min(12, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str

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
    n_barriers = 0
    for i in 4:n_lines
        line = lines[i]
        if !contains(line, "barrier")
            push!(gates, parse_gate(line))
        else
            n_barriers += 1
        end
    end
    println("Circuit built, there were $n_barriers barriers")
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

#Initial operator with "Z" on designated sites and identity everywhere else. Use sqrt(2) normalisation So Tr(O ODAG) = 1
function initial_state(sphysical, sauxillary, z_vertices)
    vs = collect(vertices(sphysical))
    ψ = ITensorNetwork(sphysical; link_space = 1)
    for v in vertices(sphysical)
        phys_ind, aux_ind = sphysical[v], sauxillary[v]
        if v ∈ z_vertices
            ITensorNetworks.@preserve_graph ψ[v] = (1/sqrt(2))*ITensors.delta(phys_ind..., aux_ind...)
        else
            ITensorNetworks.@preserve_graph ψ[v] = (1/sqrt(2))*replaceinds(ITensors.op("Z", phys_ind...), prime(phys_ind),aux_ind)
        end
    end
    ψ = ITensorNetworks.insert_linkinds(ψ)
    return ψ
end

#Swap the site indices (no conjugate) on a site, effectively is a transpose of the state (no conjugation)
function swap_physical_auxillary_inds(tn::ITensorNetwork)
    tn_swap = copy(tn)
    for v in vertices(tn)
        s1, s2 = first(siteinds(tn, v)), last(siteinds(tn, v))
        ITensorNetworks.@preserve_graph tn_swap[v] = ITensors.swapind(tn[v], s1, s2)
    end
    return tn_swap
end

function main(maxdim::Int, S::Int, grad::Float64, delta::Float64)
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
    sphysical = ITN.siteinds("S=1/2", g)
    sauxillary = ITN.siteinds("S=1/2", g)
    #Sites to initial "Z" on
    z_vertices = [45, 47, 67, 69, 89, 91]

    #Initial state
    O_init = initial_state(sphysical, sauxillary, z_vertices)
    OU = copy(O_init)

    #Do a transpose and take overlap
    UO = swap_physical_auxillary_inds(OU)
    println("Initial value of Tr(OOdag) is $(inner(OU, UO; alg = "bp"))")

    #Cache of norm of state (needed to apply gates)
    OU_bpc = build_bp_cache(OU)
    #Transform circuit to itensors
    U = TN.toitensor(circuit, sphysical)

    #Apply kwargs (maximum bond dimension)
    apply_kwargs = (; maxdim = maxdim, cutoff = 1e-10, normalize = false)

    #Apply the circuit to get O. U
    t = time()
    OU, OU_bpc, errs = apply(U, OU, OU_bpc; apply_kwargs, verbose = true)
    t = time() - t
    println("Simulation O -> OU took $(t) seconds")
    println("Average gate error was $(Statistics.mean(errs))")
    println("BP Computed Final value of Frobenius norm (should be conserved w/out truncation) is $(scalar(OU_bpc))")

    println("Rough fedility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

    #Exploit symmetry and transform to get U. O
    UO = swap_physical_auxillary_inds(OU)

    #Measure signal from BP overlap (we can use loop corrections etc if needed but simple for now)
    signal = inner(OU, UO; alg = "bp")
    println("Final value of Tr(OUOUdag) is $signal")

    save_file = "/mnt/home/jtindall/ceph/Data/Algorthmic/Heisenberg/Maxdim"*string(maxdim)*"S"*string(S)*"grad"*string(grad)*"delta"*string(delta)*".npz"
    npzwrite(save_file, signal = signal, errs = errs)
end

χ, S, grad, delta = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4])
#χ, S, grad, delta = 16, 3, 0.05, 0.011
main(χ, S, grad, delta)