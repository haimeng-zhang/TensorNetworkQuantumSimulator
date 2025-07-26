using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs: NamedGraphs
using Statistics

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

# apply orbital rotations: XX + YY gates followed by phase gates
xxyy_layer = [("R")]
# apply diagonal Coulomb evolution

χ = 8
apply_kwargs = (; cutoff = 1e-12, maxdim = χ)

# define initial state
ψt = ITensorNetwork(v -> "↑", s)
#BP cache for norm of the network
ψψ = build_bp_cache(ψt)

# evolve the state
layer = hf_layer
ψt, ψψ, errs = apply(layer, ψt, ψψ; apply_kwargs)
fidelity = prod(1.0 .- errs)
nsamples = 100
bitstrings = TN.sample_directly_certified(ψt, nsamples; norm_message_rank = 8)

# now I have the bitstring, how do I check if it is correct?
# view count distribution

# measure expectation values

# I don't know yet what these line of code is doing
# st_dev = Statistics.std(first.(bitstrings))
# println("Standard deviation of p(x) / q(x) is $(st_dev)")
