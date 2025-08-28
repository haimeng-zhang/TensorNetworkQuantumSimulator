using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics
using NamedGraphs: AbstractNamedGraph

include("../utils.jl")

using Random
using Serialization

using JLD2

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using ITensors
const IT = ITensors

using Dictionaries: Dictionary, set!

BLAS.set_num_threads(min(12, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

IT.disable_warn_order()
f_root = "/mnt/home/jtindall/ceph/Data/DWave/"

function analyse_loop_errors()

    chi =8
    #ns = [(3,3,8), (4,4,8), (5,5,8), (8,8,8), (8,8,12), (8,8,16), (10,10,12), (10,10,16), (12,12,16), (10,10,24)]
    ns = [(3,3,8), (4,4,8), (5,5,8), (8,8,12), (8,8,16), (10,10,12), (10,10,16), (12,12,16), (10,10,24)]
    boundary = "pbc"
    annealing_time = 15
    disorders = [i for i in 1:5]
    wv_folder = boundary == "obc" ? "OBCDiamond" : "Diamond"

    qubit_counts = [Int(nx*ny*0.25*nz) for (nx,ny,nz) in ns]
    loop_errs = []
    for (nx,ny,nz) in ns
        loop_err = 0
        try
            for disorder in disorders
                f_name = f_root * "Wavefunctions/"*wv_folder*"/nx"*string(nx)*"ny"*string(ny)*"nz"*string(nz)*"Chi"*string(chi)*"DisorderNo"*string(disorder)*"AnnealingTime"*string(annealing_time)*".jld2"
                f = jldopen(f_name)
                loop_errors = f["LoopErrors"]
                loop_err += Statistics.mean(loop_errors)
            end
        catch
            continue
        end
        println("Nx is $(nx), Ny is $(ny), Nz is $(nz) and loop error is $(loop_err / length(disorders))")
        push!(loop_errs, loop_err / length(disorders))
    end

    @show qubit_counts
    @show loop_errs


    @show [(qc, ge) for (qc, ge) in zip(qubit_counts, loop_errs)]
end

function analyse_gate_errors()

    save_root = "/mnt/home/jtindall/ceph/Data/DWave/PaperData/ResubmissionData/SMErrorAnalysisFigure/GateErrors"
    chi =8
    ns = [(3,3,8), (4,4,8), (5,5,8), (8,8,8), (8,8,12), (8,8,16), (10,10,12), (10,10,16), (12,12,16), (10,10,24), (12,12,24)]
    #ns = [(3,3,8), (4,4,8), (5,5,8), (8,8,12)]
    boundary = "pbc"
    annealing_time =15
    disorders = [1,2,3,4,5]
    wv_folder = boundary == "obc" ? "OBCDiamond" : "Diamond"

    qubit_counts = [Int(nx*ny*0.25*nz) for (nx,ny,nz) in ns]
    @show qubit_counts
    gate_errs = []
    for (nx,ny,nz) in ns
        gate_err = 0
        for disorder in disorders
            f_name = f_root * "Wavefunctions/"*wv_folder*"/nx"*string(nx)*"ny"*string(ny)*"nz"*string(nz)*"Chi"*string(chi)*"DisorderNo"*string(disorder)*"AnnealingTime"*string(annealing_time)*".jld2"
            f = jldopen(f_name)
            gate_errors = f["Errors"]
            #gate_err += Statistics.mean(gate_errors)
            _gate_err = 1.0 - (prod(1.0 .- gate_errors))^(1/length(gate_errors))
            gate_err += 1.0 - (prod(1.0 .- gate_errors))^(1/length(gate_errors))
            #gate_err += exp((1/length(gate_errors))*sum(log.(gate_errors)))
            npzwrite(save_root*"/nx"*string(nx)*"ny"*string(ny)*"nz"*string(nz)*"Chi"*string(chi)*"DisorderNo"*string(disorder)*"AnnealingTime"*string(annealing_time)*".npz", gate_error = _gate_err)
        end
        push!(gate_errs, gate_err / length(disorders))
    end

    @show [(qc, ge) for (qc, ge) in zip(qubit_counts, gate_errs)]
end

function analyse_time_taken()

    chi = 16
    #ns = [(3,3,8), (4,4,8), (5,5,8), (6,6,8), (8,8,8), (8,8,12), (8,8,16), (10,10,12), (10,10,16), (12,12,12), (12,12,16), (10,10,24), (12,12,24), (16,16,16)]
    ns = [(3,3,8), (4,4,8), (5,5,8), (8,8,8), (8,8,12), (8,8,16)]
    boundary = "pbc"
    annealing_time =11
    disorders = [i for i in 1:5]
    wv_folder = boundary == "obc" ? "OBCDiamond" : "Diamond"

    qubit_counts = [Int(nx*ny*0.25*nz) for (nx,ny,nz) in ns]
    @show qubit_counts
    times = []
    for (nx,ny,nz) in ns
        time_taken = 0
        for disorder in disorders
            f_name = f_root * "Wavefunctions/"*wv_folder*"/nx"*string(nx)*"ny"*string(ny)*"nz"*string(nz)*"Chi"*string(chi)*"DisorderNo"*string(disorder)*"AnnealingTime"*string(annealing_time)*".jld2"
            f = jldopen(f_name)
            time_taken += f["TimeTaken"]
        end
        push!(times, time_taken / length(disorders))
    end

    @show qubit_counts
    @show times
    @show [t/q for (t,q) in zip(times, qubit_counts)]
end

function analyse_disorders()
    chi = 16
    nx, ny, nz = 8,8,12
    boundary = "pbc"
    annealing_time =7
    disorders = [i for i in 1:5]
    wv_folder = boundary == "obc" ? "OBCDiamond" : "Diamond"

    for disorder in disorders
        f_name = f_root * "Wavefunctions/"*wv_folder*"/nx"*string(nx)*"ny"*string(ny)*"nz"*string(nz)*"Chi"*string(chi)*"DisorderNo"*string(disorder)*"AnnealingTime"*string(annealing_time)*".jld2"
        f = jldopen(f_name)
        gate_errors = f["Errors"]
        e_gate = 1.0 - (prod(1.0 .- gate_errors))^(1/length(gate_errors))
        println("Average gate error is $(e_gate)")

        loop_errors = f["LoopErrors"]
        println("Average loop error is $(Statistics.mean(loop_errors))") 

        loop_errors = f["LoopErrors"]
        println("Maximum loop error is $(Statistics.maximum(loop_errors))") 
    end
end

#analyse_loop_errors()
analyse_gate_errors()
#analyse_time_taken()
#analyse_disorders()