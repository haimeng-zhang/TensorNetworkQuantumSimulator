using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using ITensorNetworks: AbstractBeliefPropagationCache, IndsNetwork
using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices

using Random
using TOML

include("utils.jl")

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using JLD2

using Statistics
using CUDA
using Dictionaries: Dictionary, set!

using Adapt

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

function high_temperature_initial_state(sphysical, sancilla, mu, vertex_lower_half_filter)
    ψ = ITensorNetworks.random_tensornetwork(ComplexF64, sphysical; link_space = 1)
    for v in vertices(ψ)
        array = vertex_lower_half_filter(v) ? (1/2)*[1 + mu 0; 0 1 - mu] : (1/2)*[1 - mu 0; 0 1 + mu]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(ComplexF32, array, only(sphysical[v]), only(sancilla[v]))
    end

    return ITensorNetworks.insert_linkinds(ψ)
end

function sqrt_high_temperature_initial_state(sphysical, sancilla, mu, vertex_lower_half_filter)
    ψ = ITensorNetworks.random_tensornetwork(ComplexF64, sphysical; link_space = 1)
    for v in vertices(ψ)
        array = vertex_lower_half_filter(v) ? (1/sqrt(2))*[sqrt(1 + mu) 0; 0 sqrt(1 - mu)] : (1/sqrt(2))*[sqrt(1 - mu) 0; 0 sqrt(1 + mu)]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(ComplexF64, array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end

function identity_state(sphysical, sancilla)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = [1 0; 0 1]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end

function sigmaz_state(sphysical, sancilla, vz)
    ψ = ITensorNetworks.random_tensornetwork(sphysical; link_space = 1)
    for v in vertices(ψ)
        array = v == vz ? [1 0; 0 -1] : [1 0; 0 1]
        ITensorNetworks.@preserve_graph ψ[v] = ITensors.ITensor(array, only(sphysical[v]), only(sancilla[v]))
    end
    return ITensorNetworks.insert_linkinds(ψ)
end

function trace_expect(ρI::AbstractBeliefPropagationCache, obs::Vector{<:Tuple}, sphysical, sancilla)
    os = []
    for ob in obs
        op_strs, vs, coeff = TN.collectobservable(ob)
        incoming_messages = ITensorNetworks.environment(ρI, [(v, "bra") for v in vs])
        local_numer_ops = [ITensors.replaceind(ITensors.op(op_str, only(sphysical[v])), prime(only(sphysical[v])), only(sancilla[v]))  for (op_str, v) in zip(op_strs, vs)]
        ts = [incoming_messages; local_numer_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        numer = coeff * ITensors.contract(ts; sequence = seq)[]

        local_denom_ops = [ρI[(v, "bra")]  for v in vs]
        ts = [incoming_messages; local_denom_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        denom = ITensors.contract(ts; sequence = seq)[]
        push!(os, numer / denom)
    end
    return os
end

function trace_expect(tr_ρ::AbstractBeliefPropagationCache, ρ::ITensorNetwork, obs::Vector{<:Tuple}, sphysical, sancilla)
    os = []
    for ob in obs
        op_strs, vs, coeff = TN.collectobservable(ob)
        incoming_messages = ITensorNetworks.environment(tr_ρ, vs)
        local_numer_ops = [ITensors.replaceind(ITensors.op(op_str, only(sphysical[v])), prime(only(sphysical[v])), only(sancilla[v]))  for (op_str, v) in zip(op_strs, vs)]
        local_numer_ops = [local_numer_ops[i] * ρ[v] for (i, v) in enumerate(vs)]
        ts = [incoming_messages; local_numer_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        numer = coeff * ITensors.contract(ts; sequence = seq)[]

        local_denom_ops = [ρ[v] * delta(only(sphysical[v]), only(sancilla[v])) for (i, v) in enumerate(vs)]
        ts = [incoming_messages; local_denom_ops]
        seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
        denom = ITensors.contract(ts; sequence = seq)[]
        push!(os, numer / denom)
    end
    return os
end

function form_tr_ρ(ρ::ITensorNetwork, sphysical, sancilla)
    tr_ρ = copy(ρ)
    for v in vertices(ρ)
        ITensorNetworks.@preserve_graph tr_ρ[v] = ρ[v] * delta(only(sphysical[v]), only(sancilla[v]))
    end
    return tr_ρ
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obs::Tuple; use_gpu = false)
    op_vec, vs, coeff = TN.collectobservable(obs)

    ρOρ = copy(ρρ)
    for (i, v) in enumerate(vs)
        if use_gpu
            ITensorNetworks.@preserve_graph ρOρ[(v,"operator")] = CUDA.cu(ITensors.op("Id", only(sancilla[v])) * ITensors.op(op_vec[i], only(sphysical[v])))
        else
            ITensorNetworks.@preserve_graph ρOρ[(v,"operator")] = ITensors.op("Id", only(sancilla[v])) * ITensors.op(op_vec[i], only(sphysical[v]))
        end
    end

    numerator = ITensorNetworks.region_scalar(ρOρ, [(v, "ket") for v in vs])
    denominator = ITensorNetworks.region_scalar(ρρ, [(v, "ket") for v in vs])

    return coeff * numerator / denominator
end

function TN.expect(ρρ::AbstractBeliefPropagationCache, sphysical::IndsNetwork, sancilla::IndsNetwork, obss::Vector{<:Tuple}; kwargs...)
    return [TN.expect(ρρ, sphysical, sancilla, obs; kwargs...) for obs in obss]
end


function main_heisenberg_sqrt(lattice::String, seed::Int, χ::Int, ny::Int, mu::Float64, δt::Float64, J::Float64)

    Time_step =  200

    file_name = "/mnt/home/jtindall/ceph/Data/Transport/"*lattice*"/HeisenbergPictureSqrtApproach/Results/ny"*string(ny)*"maxdim"*string(χ)*"dt"*string(δt)*"mu"*string(mu)*"J"*string(J)*"TimeStep"*string(Time_step)*".jld2"
    f = jldopen(file_name)
    bp_mags = f["bp_mags"]

    g, bottom_half_vertices = transport_graph_constructor(lattice, ny)
    rows = unique(first.(collect(vertices(g))))

    obs = [("Z", [v]) for v in collect(vertices(g))]

    vs = collect(vertices(g))
    rows = unique(first.(collect(vertices(g))))
    in_bottom_half = Int64[v ∈ bottom_half_vertices ? 1 : 0 for v in collect(vertices(g))]

    top_mag_vs_time = [sum([v ∉ bottom_half_vertices ? bp_mags[j, i+1] : 0 for (j, v) in enumerate(collect(vertices(g)))]) for i in 0:Time_step]
    init_top_mag = top_mag_vs_time[1]
    transferred_mags_vs_time = top_mag_vs_time .- init_top_mag

    println("Current BP Measured magnetisation is $(sum(bp_mags[:, Time_step+1]))")


    mags_vs_row = [Statistics.mean(bp_mags[filter(i -> first(vs[i]) == r, [i for i in 1:length(vs)]), Time_step]) for r in unique(rows)]

    unique_rows = [k + 1 for k in 1:length(mags_vs_row)]

    ρρ = f["density_matrix"]
    sphysical = f["sphysical"]
    sancilla = f["sancilla"]

    ρρ = adapt(CuArray{ComplexF64}, ρρ)
    grouping_function = v -> last(v)
    group_sorting_function  = v -> first(v)
    ρρ_bmps = TensorNetworkQuantumSimulator.BoundaryMPSCache(ρρ; message_rank = 32, group_sorting_function, grouping_function)
    ρρ_bmps = ITensorNetworks.update(ρρ_bmps; alg = "bp", maxiter = 5, message_update_alg = Algorithm("orthogonal", niters = 30, tol = 1e-8))

    bmps_mags = TN.expect(ρρ_bmps, sphysical, sancilla, obs; use_gpu = true)
    bmps_mag_top = sum([v ∉ bottom_half_vertices ? bmps_mags[j] : 0 for (j, v) in enumerate(collect(vertices(g)))])
    println("Current BMPS Measured magnetisation is $(sum(bmps_mags))")
    #println("Current BMPS Measured magnetisation transfer is $(bmps_mag_top - init_mag_top)")

    err = sum([abs(bp_mags[j, Time_step+1] - bmps_mags[j]) for j in 1:length(vs)]) / sum(abs.(bmps_mags))
    println("Relative error is $(err)")
    println("Total is $(sum(abs.(bmps_mags)))")

    #_positions = (unique_rows - y0) / np.sqrt(i * deltat)

    #erf_profile = erf_propfile_y0_fixed(y0 = 0)
    #popt, pcov = curve_fit(erf_profile, _positions[1:(len(unique_rows)- 1)], mags_vs_row[1:(len(unique_rows)- 1)], p0 = [1.0, 1.0])
    #A_fit, w_fit = popt
    #D_fit = (w_fit^2 / 4)
    #print("Diffusion constant is " + str(D_fit))
end



#mode, lattice, χ, ny, mu, δt, J, seed = ARGS[1], ARGS[2], parse(Int64, ARGS[3]), parse(Int64, ARGS[4]), parse(Float64, ARGS[5]), parse(Float64, ARGS[6]), parse(Float64, ARGS[7]), parse(Int64, ARGS[8])
mode, lattice, χ, ny, mu, δt, J, seed = "HeisenbergSqrt", "Square", 8, 40, 0.05, 0.05, 1.0, 1
mode == "HeisenbergSqrt" && main_heisenberg_sqrt(lattice, seed, χ, ny, mu, δt, J)
