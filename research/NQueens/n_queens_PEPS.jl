using ITensors
using ITensorMPS

using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks: siteinds, ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str

using Random

using EinExprs: Greedy

using Base.Threads
using MKL
using LinearAlgebra

using NPZ

BLAS.set_num_threads(min(12, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

function spinup_projector(subvertices, s; cutoff, include_zero_space = false)
    H = OpSum()
    if include_zero_space
        H += prod([Op("ProjDn", i) for i in 1:length(subvertices)])
    end
    for (i, v) in enumerate(subvertices)
        H += Op("ProjUp", i) * prod([Op("ProjDn", j) for j in setdiff([k for k in 1:length(subvertices)], [i])])
    end
    siteinds = ITensorMPS.siteinds("S=1/2", length(subvertices))
    for (i, v) in enumerate(subvertices)
        siteinds[i] = only(s[v])
    end
    return truncate(MPO(H, siteinds; cutoff))
end

function project_to_spinup(ψ, subvertices, s; cutoff, include_zero_space = false)
    P = spinup_projector(subvertices, s; cutoff, include_zero_space)
    for (i, v) in enumerate(subvertices)
        ψ[v] = noprime(P[i] * ψ[v])
    end
    return ψ
end

function build_queenstate(s, n::Int; cutoff, pre_placed_queen_positions = [])

    ψ = ITensorNetworks.ITensorNetwork(v -> v ∉ pre_placed_queen_positions ? [1.0, 1.0] : [1.0, 0.0], s, link_space = nothing)

    for i in 1:n
        ψ = project_to_spinup(ψ, [(i, col) for col in 1:n], s; cutoff)
        ψ = project_to_spinup(ψ, [(row, i) for row in 1:n], s; cutoff)
    end

    for rowpluscolumn in 3:(2*n - 1)
        subvertices = filter(pos -> (first(pos)+ last(pos)) == rowpluscolumn, collect(vertices(s)))
        ψ = project_to_spinup(ψ, subvertices, s; cutoff, include_zero_space = true)
    end

    for rowminuscolumn in (2 - n):(n - 2)
        subvertices = filter(pos -> (first(pos) - last(pos)) == rowminuscolumn, [(i,j) for i in 1:n for j in 1:n])
        ψ = project_to_spinup(ψ, subvertices, s; cutoff, include_zero_space = true)
    end

    return ψ
end


function queen_graph(n::Int64)
    g = named_grid((n,n))
    for i in 1:(n-1)
        for j in 1:(n-1)
            g = NamedGraphs.GraphsExtensions.add_edge(g, NamedEdge((i,j) => (i+1, j+1)))
        end
    end

    for i in 2:(n)
        for j in 1:(n-1)
            g = NamedGraphs.GraphsExtensions.add_edge(g, NamedEdge((i,j) => (i-1, j+1)))
        end
    end
    return g
end

function project_state(ψ)
    ψ = copy(ψ)
    for v in vertices(ψ)
        ψ[v] *= ITensor(1.0, only(siteinds(ψ, v)))
    end
    return ψ
end

function _inner(M1::MPS, M2::MPS)
    ts = [[M1[i] for i in 1:length(M1)]; [M2[i] for i in 1:length(M2)]]
    seq = ITensorNetworks.contraction_sequence(ts; alg = "einexpr", optimizer = Greedy())
    return ITensors.contract(ts; sequence = seq)[]
end

function _inner(M1::MPS, C::MPO, M2::MPS)
    ts = [[M1[i] for i in 1:length(M1)]; [M2[i] for i in 1:length(M2)]; [C[i] for i in 1:length(C)]]
    seq = ITensorNetworks.contraction_sequence(ts; alg = "einexpr", optimizer = Greedy())
    return ITensors.contract(ts; sequence = seq)[]
end

function _apply_irregular_length(O::MPO, M::MPS; kwargs...)
    n = minimum((length(O), length(M)))
    ts = [O[i] *M[i] for i in 1:n]
    if length(O) < length(M)
        ts = vcat(ts, [M[i] for i in (length(O)+1):length(M)])
    elseif length(M) < length(O)
        ts = vcat(ts, [O[i] for i in (length(M)+1):length(O)])
    end
    return truncate(ITensorMPS.MPS(ts); kwargs...)
end

function contract_projected_queens_state_diagonally(ψ, n::Int; kwargs...)
    bottom_left = [(1, 1)]
    ψL = ITensorMPS.MPS([ψ[v] for v in bottom_left])
    nm = 1
    for rowpluscolumn in 3:(2*n - 1)
        diagonal = filter(v-> sum(v) == rowpluscolumn, collect(vertices(ψ)))
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in diagonal]); kwargs...)
        ψL = _apply_irregular_length(ψC, ψL; kwargs...)
        nm *= norm(ψL)
        ψL = normalize(ψL)
        @show ITensorMPS.maxlinkdim(ψL)
    end

    z = ITensorMPS.contract(vcat([ψ[(n,n)]], [ψL[i] for i in 1:length(ψL)]); sequence = "automatic")[]
    return z *nm
end

function contract_projected_queens_state_diagonally_bp(ψ, n::Int; cutoff, maxdim)
    bottom_left = [(1, 1)]
    ψL = ITensorMPS.MPS([ψ[v] for v in bottom_left])
    ms = Dictionary()
    set!(ms, 2=>3, ψL)
    for (i, rowpluscolumn) in enumerate(3:(2*n - 1))
        diagonal = filter(v-> sum(v) == rowpluscolumn, collect(vertices(ψ)))
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in diagonal]); cutoff)
        ψL = _apply_irregular_length(ψC, ψL; cutoff, maxdim)
        ψL = normalize(ψL)
        set!(ms, rowpluscolumn=>rowpluscolumn+1, ψL)
        @show ITensorMPS.maxlinkdim(ψL)
    end


    bottom_right = [(n,n)]
    ψR = ITensorMPS.MPS([ψ[v] for v in bottom_right])
    set!(ms, 2*n=>2*n-1, ψR)

    numer_terms = log(_inner(ms[2n-1=>2n], ψR))
    denom_terms = log(_inner(ms[2n-1=>2n],ms[2n=>2n-1]))

    for (i, rowpluscolumn) in enumerate((2*n - 1):-1:3)
        diagonal = filter(v-> sum(v) == rowpluscolumn, collect(vertices(ψ)))
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in diagonal]); cutoff)
        ψR = _apply_irregular_length(ψC, ψR; cutoff, maxdim)
        ψR = normalize(ψR)
        set!(ms,rowpluscolumn=>rowpluscolumn-1, ψR)
        numer_terms += log(_inner(ms[rowpluscolumn+1=>rowpluscolumn], ψC, ms[rowpluscolumn-1=>rowpluscolumn]))
        denom_terms += log(_inner(ms[rowpluscolumn=>rowpluscolumn-1], ms[rowpluscolumn-1=>rowpluscolumn]))
        @show ITensorMPS.maxlinkdim(ψR)
    end

    numer_terms += log(ITensorMPS.contract(vcat([ψ[(1,1)]], [ψR[i] for i in 1:length(ψR)]); sequence = "automatic")[])
    return numer_terms, denom_terms
end

function contract_projected_queens_state_lr(ψ, n::Int; cutoff, maxdim)
    left_col = [(row, 1) for row in 1:n]
    ψL = truncate(ITensorMPS.MPS([ψ[v] for v in left_col]); cutoff)
    nm = 1
    for i in 2:(n-1)
        col = [(row, i) for row in 1:n]
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in col]); cutoff)
        ψL_tensors = [ψL[j] * ψC[j] for j in 1:n]
        ψL = truncate(ITensorMPS.MPS(ψL_tensors); cutoff, maxdim)
        nm *= sqrt(norm(ψL))
        ψL = normalize(ψL)
        @show ITensorMPS.maxlinkdim(ψL)
    end
    right_col = [(row, n) for row in 1:n]
    ψR = truncate(ITensorMPS.MPS([ψ[v] for v in right_col]); cutoff)

    return _inner(ψL, ψR) * n
end

function contract_projected_queens_state_bp(ψ, n::Int; kwargs...)
    left_col = [(row, 1) for row in 1:n]
    ms = Dictionary()
    ψL = truncate(ITensorMPS.MPS([ψ[v] for v in left_col]); kwargs...)
    set!(ms, 1 => 2, ψL)
    for i in 2:(n-1)
        col = [(row, i) for row in 1:n]
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in col]); kwargs...)
        ψL_tensors = [ψL[j] * ψC[j] for j in 1:n]
        ψL = normalize(truncate(ITensorMPS.MPS(ψL_tensors); kwargs...))
        set!(ms, i => i + 1, ψL)
        @show ITensorMPS.maxlinkdim(ψL)
    end
    right_col = [(row, n) for row in 1:n]
    ψR = truncate(ITensorMPS.MPS([ψ[v] for v in right_col]); kwargs...)
    z = _inner(ms[n-1 => n], ψR)
    set!(ms, n => n-1, ψR)

    for i in (n-1):-1:2
        col = [(row, i) for row in 1:n]
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in col]); kwargs...)
        ψR_tensors = [ψR[j] * ψC[j] for j in 1:n]
        ψR = normalize(truncate(ITensorMPS.MPS(ψR_tensors); kwargs...))
        set!(ms, i => i - 1, ψR)
        @show ITensorMPS.maxlinkdim(ψR)
    end

    left_col = [(row, 1) for row in 1:n]
    ψL = truncate(ITensorMPS.MPS([ψ[v] for v in left_col]); kwargs...)
    z *= _inner(ms[2 => 1], ψL)

    for i in 2:(n-1)
        col = [(row, i) for row in 1:n]
        ψC = truncate(ITensorMPS.MPO([ψ[v] for v in col]); kwargs...)
        ψL, ψR = ms[i-1=>i], ms[i+1=>i] 
        z *= _inner(ψL, ψC, ψR)
    end

    denom = 1
    for pair in [i => i+1 for i in 1:(n-1)]
        denom *= _inner(ms[pair], ms[reverse(pair)])
    end

    return z / denom
end
            
function main(n::Int, maxdim::Int)  
    
    ITensors.disable_warn_order()
    cutoff = 1e-10
    g = queen_graph(n)
    s = siteinds("S=1/2", g)

    pre_placed_queen_positions = []
    println("Solving n queens for n is $(n) and maxdim $(maxdim)")
    ψ = build_queenstate(s, n; cutoff, pre_placed_queen_positions)
    ψ = project_state(ψ)
    println("State built and projected")

    flush(stdout)

    @show ITensorNetworks.maxlinkdim(ψ)

    @time numer_terms, denom_terms = contract_projected_queens_state_diagonally_bp(ψ, n; cutoff, maxdim)
    no_solutions = exp(numer_terms - denom_terms)
    println("Counted $no_solutions solutions")

    flush(stdout)

    save_file = "/mnt/home/jtindall/ceph/Data/NQueens/TensorsN"*string(n)*"maxdim"*string(maxdim)*".npz"
    npzwrite(save_file, no_solutions = no_solutions, numer_terms, denom_terms)

end

#n, maxdim = 4,8
n, maxdim = parse(Int64, ARGS[1]), parse(Int64, ARGS[2])
main(n, maxdim)
            
