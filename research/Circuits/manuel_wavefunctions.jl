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

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

function main()
    f = "/mnt/home/jtindall/ceph/Data/ManuelWavefunctions/wavefunction_willow_muInf_Delta1.0_dt0.1_maxdim20_layer15.jld2"

    f = deserialize(f)
    ψ_cpu = last(f)
    

    #ψψ = build_normsqr_bp_cache(ψ)

    vs = collect(vertices(ψ_cpu))



    println("Gauged and normalized")

    Rs = [3,5,10,15,20,30]

    for R in Rs
        CUDA.reclaim()

        ψ = CUDA.cu(ψ_cpu)
        ψ, ψψ = TN.symmetric_gauge(ψ)
        ψ, ψψ = TN.normalize(ψ, ψψ)
        println("Gauged and normalized")

        println("Running boundary mps")
        t1 = time()
        ψψ_bmps = TN.build_normsqr_bmps_cache(ψψ, R)
        t2 = time()
        println("Took $(t2 - t1) secs")

        z_vs = [vs[i] for i in [51,52,53,54,55,56,57]]

        zs_expect = TN.expect(ψψ_bmps, [("Z", [v]) for v in z_vs])

        f = "/mnt/home/jtindall/ceph/Data/ManuelWavefunctions/willow_muInf_Delta1.0_dt0.1_maxdim20_layer15_GPUexpectsR"*string(R)*".npz"
        npzwrite(f, zs_expect = zs_expect, time = t1 = t2)
    end
end

main()