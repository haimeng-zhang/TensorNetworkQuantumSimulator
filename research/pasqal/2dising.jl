using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

using CUDA

using NPZ

using ITensors: Algorithm

function main(n::Int, hx::Number, χ::Int, χ_trunc::Int, RMPS::Int)
    nx = n
    ny = n

    half_n = round(Int, n/2)
    # the graph is your main friend. This will be the geometry of the TN you wull work with
    g = named_grid((nx, ny))
    nq = length(vertices(g))

    dt = 0.01

    J = 1.0

    #Build a layer of the circuit. Pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("Rx", [v], hx*dt) for v in vertices(g))

    #For two site gates do an edge coloring to Trotterise the circuit
    ec = edge_color(g, 4)
    for colored_edges in ec
        append!(layer, ("Rzz", pair, 2*J*dt) for pair in colored_edges)
    end

    append!(layer, ("Rx", [v], hx*dt) for v in vertices(g))

    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    sz_obs = [("Z", [(half_n, half_n + i)]) for i in 0:half_n]  # right in the middle
    szz_obs = [("ZZ", [(half_n, half_n), (half_n, half_n + i)]) for i in 1:half_n]
    # the number of circuit layers
    nl = 250

    measure_steps = 5

    #The inds network
    s = siteinds("S=1/2", g)
    # the initial state (all up, use Float 32 precision)
    ψ0 = ITensorNetwork(ComplexF32, v -> "↓", s)

    #ψ0 = CUDA.cu(ψ0)

    #We are going to be doing small time steps, so we don't need to do too much BP (hence low maxiter)
    bp_update_kwargs = (; maxiter=5, tol=1e-4, message_update_alg = Algorithm("posdef_contract"))

    # max bond dimension for the TN
    apply_kwargs = (maxdim = χ, cutoff = 1e-10, normalize_tensors = true)

    # create the BP cache representing the square of the tensor network
    ψψ = build_normsqr_bp_cache(ψ0; cache_update_kwargs = bp_update_kwargs)

    # an array to keep track of expectations taken via two methods
    sz_expectations_boundarymps = zeros(Float64, (round(Int, nl / measure_steps) + 1, length(sz_obs)))
    szz_expectations_boundarymps = zeros(Float64, (round(Int, nl / measure_steps) + 1, length(szz_obs) + 1))
    correlators_boundarymps = zeros(Float64, (round(Int, nl / measure_steps) + 1, length(szz_obs) + 1))
    szz_expectations_boundarymps[1, :] = [1.0 for i in 1:(length(szz_obs)+1)]
    sz_expectations_boundarymps[1, :] = [-1.0 for i in 1:length(sz_obs)]


    simulation_time = time()

    times = [0.0]
    errs = Float64[]
    t = 0
    i = 2
    # evolve! (First step takes long due to compilation)
    for l = 1:nl
        println("Layer $l")

        t1 = @timed ψψ, errors =
            apply_gates(layer, ψψ; apply_kwargs, bp_update_kwargs, verbose = false);
        errs = append!(errs, errors)
        t += dt

        if l % measure_steps == 0
            ψ = ket_network(ψψ)

            if χ_trunc < ITensorNetworks.maxlinkdim(ψ)
                ψ_trunc, _ = TensorNetworkQuantumSimulator.symmetric_gauge(ψ; maxdim = χ_trunc, cutoff = 1e-10, cache! = Ref(ψψ))
                ψ_trunc = CUDA.cu(ψ_trunc)
            else
                ψ_trunc = CUDA.cu(ψ)
            end

            ψψ_trunc = ITensorNetworks.BeliefPropagationCache(ITensorNetworks.QuadraticFormNetwork(ψ_trunc))
            ψψ_bmps = TensorNetworkQuantumSimulator.BoundaryMPSCache(ψψ_trunc; message_rank = RMPS)

            #As we only measure things in one row, we only need to get all messages pointing towards it (half of all possible MPS messages)
            edge_seq = vcat([n - i + 1 => n-i for i in 1:(half_n)], [i => i+1 for i in 1:(half_n-1)])
            ψψ_bmps = ITensorNetworks.update(ψψ_bmps; alg = "bp", edge_sequence = edge_seq)

            sz_boundarymps = expect(ψψ_bmps,sz_obs)
            szzs_boundarymps = expect(ψψ_bmps,szz_obs)
            println("    Took time: $(t1.time) [s]. Max bond dimension: $(maxlinkdim(ψ))")
            println("    Maximum Gate error for layer was $(maximum(errors))")
            println("    Boundary MPS Measured Sigmaz is $(first(sz_boundarymps))")
            println("    Boundary MPS Measured Sigmazz is $(first((szzs_boundarymps)))")
            sz_expectations_boundarymps[i, :] = real.(sz_boundarymps)
            szz_expectations_boundarymps[i, :] = vcat([1.0], real.(szzs_boundarymps))
            correlators_boundarymps[i, :] = [szz_expectations_boundarymps[i, j] - sz_expectations_boundarymps[i, 1]*sz_expectations_boundarymps[i,j] for j in 1:length(szz_expectations_boundarymps[i, :])]
            push!(times, t)
            CUDA.reclaim()
            i += 1
        end
    end

    #Simulation time (secs)
    simulation_time = time() - simulation_time
    #Wavefunction memory (GB)
    wf_mem = Base.summarysize(ket_network(ψψ)) / 1e9

    save_file = "/mnt/home/jtindall/ceph/Data/Pasqal/2DIsingnx$(nx)ny$(ny)Quenchh$(hx)Chi$(χ)ChiTrunc$(χ_trunc)RMPS$(RMPS)Expects.npz"
    npzwrite(save_file, wf_mem = wf_mem, simulation_time = simulation_time, sz_expectations_boundarymps = sz_expectations_boundarymps, szz_expectations_boundarymps = szz_expectations_boundarymps, correlators_boundarymps = correlators_boundarymps, errs = errs, times = times)


end
# n =parse(Int64, ARGS[1])
# hx =parse(Float64, ARGS[2])
# χ =parse(Int64, ARGS[3])
# χ_trunc =parse(Int64, ARGS[4])
# RMPS =parse(Int64, ARGS[5])
n =10
hx =0.5
χ =40
χ_trunc =24
RMPS =32

main(n, hx, χ, χ_trunc, RMPS)
