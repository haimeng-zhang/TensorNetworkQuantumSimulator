using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

using NPZ

using JLD2

function main(Δ::Number, χ::Int)
    nx = 14
    ny = 14

    # the graph is your main friend. This will be the geometry of the TN you wull work with
    g = named_grid((nx, ny))

    dt = 0.157
    Ω = 1.0
    Rb_a = 1.4
    Rb_a_six = (Rb_a^6)

    #Build a layer of the circuit. Pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("Rx", [v], 0.5*Ω*dt) for v in vertices(g))
    append!(layer, ("Rz+", [v], -Δ*dt) for v in vertices(g))

    #For two site gates do an edge coloring to Trotterise the circuit
    ec = edge_color(g, 4)
    for colored_edges in ec
        append!(layer, ("Rz+z+", pair, 2*((Rb_a_six))*dt) for pair in colored_edges)
    end

    append!(layer, ("Rz+", [v], -Δ*dt) for v in vertices(g))
    append!(layer, ("Rx", [v], 0.5*Ω*dt) for v in vertices(g))

    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    obs = [("ProjDn", [v]) for v in vertices(g)]  # right in the middle

    # the initial state (all up, use Float 32 precision)
    ψ0 = zerostate(ComplexF32,g)

    # max bond dimension for the TN
    apply_kwargs = (maxdim = χ, cutoff = 1e-10, normalize_tensors = true)

    # create the BP cache representing the square of the tensor network
    ψ_bpc = BeliefPropagationCache(ψ0)

    # an array to keep track of expectations taken via two methods
    expectations_bmps = [0.0]
    expectations_bp = [0.0]
    bp_times, bmps_times = [0.0], [0.0]

    mps_bond_dimension = 2*χ
    nl = 1000

    bp_measurement_time = 10
    bmps_measurement_time =250

    # evolve! (First step takes long due to compilation)
    for l = 1:nl

        t1 = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false, transfer_to_gpu = false);

        if l % bp_measurement_time == 0
            println("Layer $l")
            sz_bp = expect(ψ_bpc, obs)
            println("    Took time: $(t1.time) [s]. Max bond dimension: $(maxlinkdim(ψ_bpc))")
            println("    Maximum Gate error for layer was $(maximum(errors))")

            println("    BP Measured Sigmaz is $(Statistics.mean(sz_bp))")
            push!(expectations_bp, real(Statistics.mean(sz_bp)))
            push!(bp_times, 0.01*nl)
        end

        if l % bmps_measurement_time == 0
            # sz_bmps = expect(CUDA.cu(network(ψ_bpc)), obs; alg = "boundarymps", mps_bond_dimension)
            # println("    BMPS Measured Sigmaz is $(Statistics.mean(sz_bmps))")
            # push!(expectations_bmps, real(Statistics.mean(sz_bmps)))
            # push!(bmps_times, 0.01*nl)
            jldsave("/mnt/home/jtindall/ceph/Data/QuEra/Wavefunctions/nx14ny14WfDelta"*string(Δ)*"Chi"*string(χ)*"Time"*string(l)*".npz", wavefunction = network(ψ_bpc))
        end
    end

    npzwrite("/mnt/home/jtindall/ceph/Data/QuEra/Measurements/nx14ny14Delta"*string(Δ)*"Chi"*string(χ)*".npz", bmps_times = bmps_times, bp_times = bp_times, expectations_bmps = expectations_bmps, expectations_bp = expectations_bp)
end

Δ = parse(Float64, ARGS[1])
χ = parse(Int, ARGS[2])

#Δ = 2.0
#χ = 4
main(Δ, χ)
