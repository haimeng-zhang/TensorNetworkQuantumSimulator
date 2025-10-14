using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics
using Random
using Serialization

using JLD2

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using ITensors: ITensors, Algorithm
const IT = ITensors

using Dictionaries: Dictionary, set!

using CUDA

using Adapt

BLAS.set_num_threads(min(1, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

function sigmazs(probs_and_bitstrings, vs)
    os = [0 for v in vs]
    for (x,y,bitstring) in probs_and_bitstrings
        for (i, v) in enumerate(vs)
            os[i] += (-2*bitstring[v]+1)
        end
    end

    return os ./ length(probs_and_bitstrings)
end

function main()
    f = "/mnt/home/jtindall/ceph/Data/ManuelWavefunctions/wavefunction_willow_muInf_Delta1.0_dt0.1_maxdim20_layer15.jld2"

    f = deserialize(f)
    ψ_cpu = last(f)

    for v in vertices(ψ_cpu)
        ψ_cpu[v] = adapt(Vector{ComplexF32})(ψ_cpu[v])
    end

    vs = collect(vertices(ψ_cpu))

    GPU = parse(Int64, ARGS[1])
    R = parse(Int64, ARGS[2])
    #GPU = 1
    #R = 5

    _nsamples = [10, 1000]
    Rs = [3, R]

    for (k, R) in enumerate(Rs)
        println("R is $(R)")
        z_vs = [vs[i] for i in [51,52,53,54,55,56,57]]

        if GPU == 1
            println("Using GPU")
            CUDA.reclaim()
            ψ = CUDA.cu(ψ_cpu)
        else
            println("Using CPU")
            ψ = copy(ψ_cpu)
        end

        ψ, ψψ = TN.symmetric_gauge(ψ)
        #ψψ = build_normsqr_bp_cache(ψ)
        ψ, ψψ = TN.normalize(ψ, ψψ)
        println("Gauged and normalized")

        flush(stdout)

        nsamples = _nsamples[k]

        bp_expect = expect(ψψ, [("Z", [v]) for v in z_vs])

        @show bp_expect

        println("Running boundary mps")

        t1 = time()
        ψψ_bmps = TN.build_normsqr_bmps_cache(ψψ, R, cache_update_kwargs = (; maxiter = 1, message_update_alg = Algorithm("orthogonal", tolerance = 1e-7, niters = 50)))
        t2 = time()
        norm_bmps_time = t2-t1
        println("Took $(t2 - t1) secs to run Boundary MPS")

        bmps_zs_expect = TN.expect(ψψ_bmps, [("Z", [v]) for v in z_vs])

        flush(stdout)

        @show bmps_zs_expect

        hardware_label = GPU == 1 ? "GPU" : "CPU"
        f = "/mnt/home/jtindall/ceph/Data/ManuelWavefunctions/willow_muInf_Delta1.0_dt0.1_maxdim20_layer15_"*hardware_label*"expectsR"*string(R)*"nosampling.npz"
        if k!= 1
            npzwrite(f, bmps_zs_expect = bmps_zs_expect, norm_bmps_time = norm_bmps_time)
        end

        #Throwaway samples
        _, _ = TN._sample(ψ, ψψ_bmps, 5; projected_message_rank = R, projected_message_update_kwargs = (;cutoff = 1e-7, maxdim = R))

        println("Collecting $(nsamples) samples")
        t1 = time()
        probs_and_bitstrings, ψ = TN._sample(ψ, ψψ_bmps, nsamples; projected_message_rank = R, projected_message_update_kwargs = (;cutoff = 1e-7, maxdim = R))
        sample_zs_expect = sigmazs(probs_and_bitstrings, z_vs)
        t2 = time()
        sample_time = t2 - t1

        println("Took $(t2 - t1) secs to sample")

        @show sample_zs_expect

        pqs = first.(probs_and_bitstrings)

        hardware_label = GPU == 1 ? "GPU" : "CPU"
        f = "/mnt/home/jtindall/ceph/Data/ManuelWavefunctions/willow_muInf_Delta1.0_dt0.1_maxdim20_layer15_"*hardware_label*"expectsR"*string(R)*".npz"
        if k != 1
            npzwrite(f, bmps_zs_expect = bmps_zs_expect, norm_bmps_time = norm_bmps_time, sample_time = sample_time, sample_zs_expect = sample_zs_expect, pqs = pqs)
        end
    end
end

main()