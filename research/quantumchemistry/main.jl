using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using NamedGraphs: NamedGraphs, neighbors
using ITensorNetworks: ITensorNetworks, ITensorNetwork
using Statistics
using JSON
using Adapt
using ITensors: Algorithm

#Map the gates in the circuit to something the simulator understands (including the vertex renaming we did)
function map_json_circuit(json_circuit, renamings)
    circuit = []
    map = Dict("p" => "P", "cp" => "CPHASE", "x" => "X", "xx_plus_yy" => "xx_plus_yy")
    for gate in json_circuit
        gate_str = first(gate)
        #Reverse vertex order to apply the xx plus yy on the right qubit? Qiskit is very ambiguous about this and I don't like it.
        gate_params = length(gate[3]) == 1 ? only(gate[3]) : gate[3]
        _gate = (map[gate_str], reverse([renamings[v + 1] for v in gate[2]]), gate_params)
        push!(circuit, _gate)
    end
    return circuit
end

#Some mapping from integer heavy hex coords to 2D coords (don't know how Manuel figured this out, so you may need to write your own as I think it only applies to 52q and 72q setups)
function getvertexmapping(initial_edges, g)
    current_edges = copy(initial_edges)

    renamings = Dict(current_edges[1] => (1, 1), current_edges[2] => (2, 1))
    for x in 2:round(Int, length(g.vertices) / 2)

        neighbors1 = neighbors(g, current_edges[1])
        new_neighbor1 = only([v for v in neighbors1 if !(v in keys(renamings))])

        neighbors2 = neighbors(g, current_edges[2])
        new_neighbor2 = only([v for v in neighbors2 if !(v in keys(renamings))])

        current_edges[1] = new_neighbor1
        current_edges[2] = new_neighbor2

        renamings[new_neighbor1] = (1, x)
        renamings[new_neighbor2] = (2, x)
    end
    back_renamings = Dict(zip(values(renamings), keys(renamings)))
    @show renamings
    return renamings, back_renamings
end

function main()
    χ = 64

    #Get the circuit
    json_circuit = JSON.parsefile("research/quantumchemistry/n2_cc-pvdz_10e26o_heavy-hex_nreps-1_random_circuit.json")
    # json_circuit = sort(json_circuit, by=g -> g[1] !== "x") # put the "x" first

    # Get the topology from the circuit (basically which qubits directly interact via gates)
    topology = unique([entry[2] .+ 1 for entry in json_circuit if length(entry[2]) == 2])
    g = TN.topologytograph(topology)


    ## Convert it to 2D co-ords (i,j) = (row, column) coordinates with edges only (i,j) => (i +-1, j) and (i,j) => (i, j+-1). This is necessary for running boundaryMPS (sampling e.g.), not gate application 
    current_edges = [1, 27]
    renamings, back_renamings = getvertexmapping(current_edges, g)
    g = NamedGraphs.rename_vertices(v -> renamings[v], g)

    #BP kwargs. The circuit is not formatted in a nice Trotterised way (its from Qiskit) so these are important as they affect the speed.
    #Also doing 32-bit precision so need a tolerance that reflects that
    bp_update_kwargs = (; maxiter=10, tol=1e-5, message_update_alg = Algorithm("squarebp"))

    # the initial state
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(ComplexF32, v -> "↑", s)

    # maximal bonddimension
    apply_kwargs = (maxdim=χ, cutoff = 1e-8, normalize_tensors = true)

    println("Applying the circuit")

    circuit = map_json_circuit(json_circuit, renamings)
    @time ψ , errors = apply_gates(circuit, ψ; bp_update_kwargs,  apply_kwargs, verbose=true)

    println("Final square fidelity estimate is $(prod(1 .- errors))")

    #Sample from q(x) and get p(x) / q(x) for each sample too
    nsamples = 10
    bitstrings = TN.sample_directly_certified(ψ, nsamples; norm_message_rank = 4)

    st_dev = Statistics.std(first.(bitstrings))
    mean = Statistics.mean(first.(bitstrings))
    println("Standard deviation of p(x) / q(x) is $(st_dev)")

    println("Mean of p(x) / q(x) is $(mean)")

    return nothing
end

main()