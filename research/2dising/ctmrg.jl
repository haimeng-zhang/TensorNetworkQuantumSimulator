using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors: datatype

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: AbstractBeliefPropagationCache

using NamedGraphs.PartitionedGraphs: PartitionEdge

using Combinatorics

using Dictionaries

using Adapt

using TensorOperations: TensorOperations, @tensor

#Store with C1 as upper left, T1 as upper
struct CTMRG_ITN
    C1::ITensor
    T1::ITensor
    centre_tensors::Vector{<:ITensor}
    centre_indexes
    boundary_indexes::Dictionary
end

function Base.copy(ctmrg::CTMRG_ITN)
    return CTMRG_ITN(copy(ctmrg.C1), copy(ctmrg.T1), copy(ctmrg.centre_tensors), ctmrg.centre_indexes, ctmrg.boundary_indexes)
end

function get_effective_environment(ctmrg::CTMRG_ITN)
    Ts = vcat([ctmrg.T1] ,[Ti(ctmrg, i) for i in 2:4])
    Cs = vcat([ctmrg.C1] ,[Ci(ctmrg, i) for i in 2:4])
    return vcat(Cs, Ts)
end

function effective_scalar(ctmrg::CTMRG_ITN; centre_tensors = ctmrg.centre_tensors)
    ts = [get_effective_environment(ctmrg); centre_tensors]
    seq = ITensorNetworks.contraction_sequence(ts; alg = "optimal")
    return ITensors.contract(ts; sequence = seq)[]
end

function environment_dim(ctmrg::CTMRG_ITN)
    return dim(ctmrg.boundary_indexes["11"])
end

function generate_boundary_indices(R::Int)
    return Dictionary(["11", "12", "22", "23", "33", "34", "44", "41"], [Index(R, "11"), Index(R, "12"), Index(R, "22"), Index(R, "23"), Index(R, "33"), Index(R, "34"), Index(R, "44"), Index(R, "41")])
end

function set_boundaryindexes!(ctmrg::CTMRG_ITN, R::Int)
    d = generate_boundary_indices(R)
    bis = boundary_indexes(ctmrg)
    for k in keys(d)
        set!(bis, k, d[k])
    end
end

function diff(ctmrga::CTMRG_ITN, ctmrgb::CTMRG_ITN)
    environment_dim(ctmrga) != environment_dim(ctmrgb) && return nothing
    E1, E2 = normalize(ctmrga.C1*ctmrga.T1), normalize(ctmrgb.C1*ctmrgb.T1)
    E1 = replaceinds(E1, [ctmrga.boundary_indexes["41"],ctmrga.boundary_indexes["12"]], [ctmrgb.boundary_indexes["41"], ctmrgb.boundary_indexes["12"]])
    return 1 - abs2(dot(E1, E2))
end

function update(ctmrg::CTMRG_ITN, niters::Int; tol = 1e-12, cutoff = 1e-12, maxdim, alg = "svd", verbose = true)
    current_ctmrg = copy(ctmrg)
    eps = nothing
    for i in 1:niters
        Q11, thinned_index = alg == "svd" ? get_Q_svd(current_ctmrg; maxdim, cutoff) : get_Q_qr(current_ctmrg)
        new_ctmrg = renormalize(current_ctmrg, Q11, thinned_index)
        eps= diff(current_ctmrg, new_ctmrg)
        if verbose && isnothing(eps)
            println("Environment Bond dimension Changed from $(environment_dim(current_ctmrg)) to $(environment_dim(new_ctmrg))")
        elseif verbose && !isnothing(tol) && eps <= tol
            println("Converged with epsilon = $(eps)")
            return new_ctmrg
        end

        current_ctmrg = new_ctmrg
    end
    !isnothing(tol) && verbose && println("Did not converge, final eps was $(eps)")
    return current_ctmrg
end

function CTMRG_ITN(centre_tensors::Vector{<:ITensor}, centre_indexes, initial_dim::Int; initialization = "Direct")
    boundary_indexes = generate_boundary_indices(initial_dim)
    if initialization == "Direct"
        T1 =denseblocks(delta([boundary_indexes["11"], boundary_indexes["12"]]))*ITensor(1.0, centre_indexes[1])
        C1 = denseblocks(delta([boundary_indexes["11"], boundary_indexes["41"]]))
    else
        T1 = random_itensor(vcat([boundary_indexes["11"], boundary_indexes["12"]], centre_indexes[1]))
        C1 = random_itensor([boundary_indexes["11"], boundary_indexes["41"]])
    end
    C1, T1 = adapt(datatype(first(centre_tensors)), C1),  adapt(datatype(first(centre_tensors)), T1)
    C1, T1 = normalize(C1), normalize(T1)
    return CTMRG_ITN(C1, T1, centre_tensors, centre_indexes, boundary_indexes)
end

function CTMRG_ITN(centre_tensors::Vector{<:ITensor}, centre_indexes, ctmrg::CTMRG_ITN)
    C1, T1 = ctmrg.C1, ctmrg.T1
    ds =[c1 != c2 ? ITensors.denseblocks(ITensors.delta(c1, c2)) : ITensor(1.0) for (c1, c2) in zip(ctmrg.centre_indexes[1], centre_indexes[1])]
    ds = [adapt(datatype(C1))(d) for d in ds]
    T1 = T1 * prod(ds)
    return CTMRG_ITN(C1, T1, centre_tensors, centre_indexes, ctmrg.boundary_indexes)
end

#Q is the isometry lives between C1 and T1 in this case
function get_Q_qr(ctmrg::CTMRG_ITN)
    Qinds = vcat([last(ctmrg.boundary_indexes)], ctmrg.centre_indexes[1])
    Q, R = ITensors.qr(ctmrg.C1 * ctmrg.T1, Qinds)
    Q = ITensors.replaceind(Q, last(ctmrg.boundary_indexes), first(ctmrg.boundary_indexes))
    Q = ITensors.replaceinds(Q, ctmrg.centre_indexes[1], ctmrg.centre_indexes[4])
    return Q, commonind(Q, R)
end

function get_Q_svd(ctmrg::CTMRG_ITN; kwargs...)
    corner_tensors = vcat(ITensor[ctmrg.T1, ctmrg.C1, Ti(ctmrg, 4)], ctmrg.centre_tensors)
    linds = vcat([ctmrg.boundary_indexes["44"]], ctmrg.centre_indexes[3])
    seq = ITensors.optimal_contraction_sequence(corner_tensors)
    C = ITensors.contract(corner_tensors; sequence = seq)
    rinds = setdiff(inds(C), linds)
    Q, D, V = ITensors.svd(C, linds; kwargs...)
    #D,Q = ITensors.eigen(C, rinds, linds; maxdim = ctmrg.R, ishermitian = true)
    thinned_index = commonind(D, Q)
    Q_mapped = replaceind(Q, ctmrg.boundary_indexes["44"], ctmrg.boundary_indexes["11"])
    Q_mapped = replaceinds(Q_mapped, ctmrg.centre_indexes[3], ctmrg.centre_indexes[4])
    return Q_mapped, thinned_index
end

function Ti(ctmrg::CTMRG_ITN, i::Int)
    bkeys = i == 2 ? ("22", "23") : i == 3 ? ("33", "34") : ("44", "41")
    Ti = copy(ctmrg.T1)
    Ti = replaceinds(Ti, ctmrg.centre_indexes[1], ctmrg.centre_indexes[i])
    Ti = replaceinds(Ti, [ctmrg.boundary_indexes["11"], ctmrg.boundary_indexes["12"]], [ctmrg.boundary_indexes[first(bkeys)], ctmrg.boundary_indexes[last(bkeys)]])
    return Ti
end

function Ci(ctmrg::CTMRG_ITN, i::Int)
    bkeys = i == 2 ? ("12", "22") : i == 3 ? ("23", "33") : ("34", "44")
    Ci = copy(ctmrg.C1)
    Ci = replaceinds(Ci, [ctmrg.boundary_indexes["11"], ctmrg.boundary_indexes["41"]], [ctmrg.boundary_indexes[first(bkeys)], ctmrg.boundary_indexes[last(bkeys)]])
    return Ci
end

function renormalize(ctmrg::CTMRG_ITN, Q11::ITensor, thinned_index::Index)
    new_bis = generate_boundary_indices(dim(thinned_index))
    Tp_tensors = vcat(ITensor[ctmrg.T1], ctmrg.centre_tensors)
    seq = ITensors.optimal_contraction_sequence(Tp_tensors)
    Tp = ITensors.contract(Tp_tensors; sequence = seq)
    Tp = Tp * dag(Q11)
    Tp = replaceind(Tp, thinned_index, new_bis["11"])
    Q12 = replaceinds(Q11, ctmrg.centre_indexes[4], ctmrg.centre_indexes[2])
    Q12 = replaceind(Q12, ctmrg.boundary_indexes["11"], ctmrg.boundary_indexes["12"])
    Tp = Tp * Q12
    Tp = replaceind(Tp, thinned_index, new_bis["12"])
    Tp = replaceinds(Tp, ctmrg.centre_indexes[3], ctmrg.centre_indexes[1])
    
    Cp_tensors = vcat(ITensor[ctmrg.T1, ctmrg.C1, Ti(ctmrg, 4)], ctmrg.centre_tensors)

    Q44 = replaceinds(Q11, ctmrg.centre_indexes[4],ctmrg.centre_indexes[3])
    Q44 = replaceind(Q44, ctmrg.boundary_indexes["11"], ctmrg.boundary_indexes["44"])
    Q44 = replaceind(Q44, thinned_index, prime(new_bis["41"]))
    Q12 = replaceind(Q12, thinned_index, prime(new_bis["11"]))
    Cp_tensors = [Cp_tensors; [dag(Q44), Q12]]
    seq = ITensors.optimal_contraction_sequence(Cp_tensors)
    Cp = ITensors.contract(Cp_tensors; sequence = seq)
    Cp = noprime(Cp)
    Cp, Tp = normalize(Cp), normalize(Tp)

    return CTMRG_ITN(Cp, Tp, ctmrg.centre_tensors, ctmrg.centre_indexes, new_bis)
end

function corner_term(ctmrg::CTMRG_ITN)
    Cs = vcat([ctmrg.C1] ,[Ci(ctmrg, i) for i in 2:4])
    ds = [ITensors.delta(ctmrg.boundary_indexes["11"],ctmrg.boundary_indexes["12"]), ITensors.delta(ctmrg.boundary_indexes["22"],ctmrg.boundary_indexes["23"]),
    ITensors.delta(ctmrg.boundary_indexes["33"],ctmrg.boundary_indexes["34"]), ITensors.delta(ctmrg.boundary_indexes["44"],ctmrg.boundary_indexes["41"])]
    ts = [Cs; ds]
    seq = ITensors.optimal_contraction_sequence(ts)
    return ITensors.contract(ts; sequence = seq)[]
end

function east_edge_term(ctmrg::CTMRG_ITN)
    Cs = vcat([ctmrg.C1] ,[Ci(ctmrg, i) for i in 2:4])
    ds = [ITensors.delta(ctmrg.boundary_indexes["11"],ctmrg.boundary_indexes["12"]), ITensors.delta(ctmrg.boundary_indexes["33"],ctmrg.boundary_indexes["34"])]
    ds = vcat(ds, [ITensors.delta(c1, c2) for (c1, c2) in zip(ctmrg.centre_indexes[2], ctmrg.centre_indexes[4])])
    Ts = [Ti(ctmrg, 2), Ti(ctmrg, 4)]
    ts = [Cs; ds; Ts]
        seq = ITensors.optimal_contraction_sequence(ts)
    return ITensors.contract(ts; sequence = seq)[]
end

function south_edge_term(ctmrg::CTMRG_ITN)
    Cs = vcat([ctmrg.C1] ,[Ci(ctmrg, i) for i in 2:4])
    ds = [ITensors.delta(ctmrg.boundary_indexes["22"],ctmrg.boundary_indexes["23"]), ITensors.delta(ctmrg.boundary_indexes["44"],ctmrg.boundary_indexes["41"])]
    ds = vcat(ds, [ITensors.delta(c1, c2) for (c1, c2) in zip(ctmrg.centre_indexes[1], ctmrg.centre_indexes[3])])
    Ts = [ctmrg.T1, Ti(ctmrg, 3)]
    ts = [Cs; ds; Ts]
        seq = ITensors.optimal_contraction_sequence(ts)
    return ITensors.contract(ts; sequence = seq)[]
end

function ITensors.scalar(ctmrg::CTMRG_ITN)
    return effective_scalar(ctmrg) * corner_term(ctmrg) / (east_edge_term(ctmrg) * south_edge_term(ctmrg))
end