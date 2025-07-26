using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs: NamedGraphs
using Statistics
using JSON

# define lattice
g = TN.heavy_hexagonal_lattice(1, 2) # a NamedGraph
# visualize

# define physical indices on each site
s = ITN.siteinds("S=1/2", g)

#Define an edge coloring
ec = edge_color(g, 3)

# define circuit parameters
norb = 8
nelec = (5, 5)

# define the circuit
# prepare hartree-fock state

alpha_nodes = [(4, 1), (5, 1), (5, 2), (5, 3), (6, 3), (7, 3), (7, 4), (7, 5)]
beta_nodes = [(1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
pairs_aa = [(a, b) for (a, b) in zip(alpha_nodes[1:end-1], alpha_nodes[2:end])]
pairs_ab =[(alpha_nodes[4], beta_nodes[4])] # this is hard coded for now

# alpha sector
occupied_orbitals = vcat(alpha_nodes[1:nelec[1]], beta_nodes[1:nelec[2]])

hf_layer = [("X", [v]) for v in occupied_orbitals]

# read gate defnitions from file
filename = "examples/lucj_n2_8o5e.json"
data = JSON.parsefile(filename)
# convert data to a layer of gates
# this is going to be the first function that I write in Julia

function format_gate_name(gate_name::String)
   mapping = Dict(
    "x" => "X",
    "xx_plus_yy" => "Rxxyy",
    "p" => "P",
    "cp" => "CPHASE",
    )
    if haskey(mapping, gate_name)
        return mapping[gate_name]
    else
        @warn "Gate name '$gate_name' not recognized, using original name"
        return gate_name
    end
end

function parse_qubit_index(qubit_index::Int64)
    mapping = Dict{Int, Tuple{Int, Int}}(
        0 => (4, 1),
        1 => (5, 1),
        2 => (5, 2),
        3 => (6, 3), 
        4 => (7, 3),
        5 => (7, 4),
        6 => (7, 5),
        7 => (6, 5),
        8 => (1, 2),
        9 => (1, 3),
        10 => (2, 3),
        11 => (3, 3),
        12 => (3, 4),
        13 => (3, 5),
        14 => (4, 5),
        15 => (5, 5),
    )
    if haskey(mapping, qubit_index)
        return mapping[qubit_index]
    else
        @warn "Qubit index '$qubit_index' not recognized, using original index"
        return qubit_index
    end   
end

function parse_gate(d::Dict{String, Any})
    name = format_gate_name(d["name"])
    qubits = Vector{Int}(d["qubits"])
    if length(qubits) == 1
        qubits = parse_qubit_index(qubits[1])
    elseif length(qubits) == 2
        qubits = (parse_qubit_index(qubits[1]), parse_qubit_index(qubits[2]))
    else
        error("Unsupported number of qubit length: $(length(qubits))")
    end
    params = d["params"]
    return (name, [qubits], params)
end

function parse_layer(data_dict::Vector; exclude_gates::Vector{String} = [])
    layer = []
    for d in data_dict
        if !isa(d, Dict)
            error("Expect Dict, got $(typeof(d))")
        end

        if d["name"] in exclude_gates
            continue
        else  # skip gates that are in the exclude list
            gate = parse_gate(d)
            push!(layer, gate)
        end
    end
    return layer
end

layer = parse_layer(data[2:end]; exclude_gates = ["global_phase", "measure", "barrier"]) # skip the first element which is qubit indices

# apply orbital rotations: XX + YY gates followed by phase gates

# apply diagonal Coulomb evolution

χ = 8
apply_kwargs = (; cutoff = 1e-12, maxdim = χ)

# define initial state
# ψt = ITensorNetwork(v -> "↑", s)
#BP cache for norm of the network
# ψψ = build_bp_cache(ψt)

# evolve the state

# layer = hf_layer
# ψt, ψψ, errs = apply(layer, ψt, ψψ; apply_kwargs)
# fidelity = prod(1.0 .- errs)
# nsamples = 100
# bitstrings = TN.sample_directly_certified(ψt, nsamples; norm_message_rank = 8)

# now I have the bitstring, how do I check if it is correct?
# view count distribution

# measure expectation values

# I don't know yet what these line of code is doing
# st_dev = Statistics.std(first.(bitstrings))
# println("Standard deviation of p(x) / q(x) is $(st_dev)")
