
using ITensors
using ITensorMPS

using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks: siteinds, ITensorNetworks, ITensorNetwork
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, datatype

using Random

using EinExprs: Greedy

using NPZ

using JLD2

using CUDA

using Adapt


solcounts = Dict(
    1  => 1,
    2  => 0,
    3  => 0,
    4  => 2,
    5  => 10,
    6  => 4,
    7  => 40,
    8  => 92,
    9  => 352,
    10 => 724,
    11 => 2680,
    12 => 14200,
    13 => 73712,
    14 => 365596,
    15 => 2279184,
    16 => 14772512,
    17 => 95815104,
    18 => 666090624,
    19 => 4968057848,
    20 => 39029188884,
    21 => 314666222712,
    22 => 2691008701644,
    23 => 24233937684440,
    24 => 227514171973736,
    25 => 2207893435808352,
    26 => 22317699616364044,
    27 => 234907967154122528,
)


function project_to_spinup(ψ, subvertices, s; include_zero_space = false)
    P = queen_projector(s, subvertices; include_zero_space)
    for (i, v) in enumerate(subvertices)
        ψ[v] = noprime(P[i] * ψ[v])
    end
    return ψ
end

function queen_projector(s, subvertices::Vector; include_zero_space = false)
    Ts = ITensor[]

    #Dn is no queen, up is a queen
    lind = Index(2, "alpha0")
    for (i,v) in enumerate(subvertices)
        rind = Index(2, "alpha"*string(i))
        sind = only(s[v])
        T = ITensor(0.0, [lind, rind, sind, sind'])
        T[lind=>1, rind=>1, sind=>2, prime(sind)=>2] = 1.0
        T[lind=>2, rind=>2, sind=>2, prime(sind)=>2] = 1.0
        T[lind=>2, rind=>1, sind=>1, prime(sind)=>1] = 1.0

        if i == 1
            T *= onehot(lind => 2)
        elseif i == length(subvertices)
            if !include_zero_space
                T *= onehot(rind => 1)
            else
                T *= ITensor(1.0, rind)
            end
        end
        push!(Ts, T)
        lind = copy(rind)
    end

    return ITensorMPS.MPO(Ts)
end

function build_queenstate(s, n::Int)

    ψ = ITensorNetworks.ITensorNetwork(v -> [1.0, 1.0], s)
    for i in 1:n
        ψ = project_to_spinup(ψ, [(i, col) for col in 1:n], s)
        ψ = project_to_spinup(ψ, [(row, i) for row in 1:n], s)
    end

    for rowpluscolumn in 3:(2*n - 1)
        subvertices = filter(pos -> (first(pos)+ last(pos)) == rowpluscolumn, collect(vertices(s)))
        ψ = project_to_spinup(ψ, subvertices, s; include_zero_space = true)
    end

    for rowminuscolumn in (2 - n):(n - 2)
        subvertices = filter(pos -> (first(pos) - last(pos)) == rowminuscolumn, [(i,j) for i in 1:n for j in 1:n])
        ψ = project_to_spinup(ψ, subvertices, s; include_zero_space = true)
    end

    return ITensorNetworks.combine_linkinds(ψ)
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

function project_state(ψ; pre_placed_queens = [], pre_placed_no_queens = [])
    ψ = copy(ψ)
    for v in vertices(ψ)
        if v ∉ pre_placed_queens && v ∉ pre_placed_no_queens
            ψ[v] *= ITensor(1.0, only(siteinds(ψ, v)))
        elseif v ∈ pre_placed_queens
            ψ[v] *= ITensors.onehot(only(siteinds(ψ, v)) => 1)
        else
            ψ[v] *= ITensors.onehot(only(siteinds(ψ, v)) => 2)
        end
    end
    return ψ
end



function contract_projected_queens_state_bp(ψ, n::Int; kwargs...)
    left_col = [(row, 1) for row in 1:n]
    ms = Dictionary()
    ψL = ITensorMPS.MPS([ψ[v] for v in left_col])
    set!(ms, 1 => 2, adapt(Vector{Float64})(ψL))
    numers = ComplexF64[]
    for i in 2:(n-1)
        col = [(row, i) for row in 1:n]

        ψL = ITensorMPS.apply(ITensorMPS.MPO([ψ[v] for v in col]), ψL; kwargs...)
        set!(ms, i => i + 1, adapt(Vector{Float64})(ψL))
        @show ITensorMPS.maxlinkdim(ψL)
    end
    ψL = nothing
    GC.gc()
    right_col = [(row, n) for row in 1:n]
    ψR = ITensorMPS.MPS([ψ[v] for v in right_col])
    push!(numers, inner(adapt(CuArray{Float64}, ms[n-1 => n]), ψR))
    set!(ms, n => n-1, adapt(Vector{Float64})(ψR))

    for i in (n-1):-1:2
        col = [(row, i) for row in 1:n]
        ψR = ITensorMPS.apply(ITensorMPS.MPO([ψ[v] for v in col]), ψR; kwargs...)
        set!(ms, i => i - 1, adapt(Vector{Float64})(ψR))
        @show ITensorMPS.maxlinkdim(ψR)
    end

    ψR = nothing
    GC.gc()

    left_col = [(row, 1) for row in 1:n]
    ψL = ITensorMPS.MPS([ψ[v] for v in left_col])
    push!(numers, inner(adapt(CuArray{Float64}, ms[2 => 1]), ψL))

    for i in 2:(n-1)
        col = [(row, i) for row in 1:n]
        push!(numers, inner(adapt(CuArray{Float64}, ms[i-1 => i]), ITensorMPS.MPO([ψ[v] for v in col]), adapt(CuArray{Float64}, ms[i+1 => i])))
    end

    denoms = ComplexF64[]
    for pair in [i => i+1 for i in 1:(n-1)]
        push!(denoms, inner(adapt(CuArray{Float64}, ms[pair]), adapt(CuArray{Float64}, ms[reverse(pair)])))
    end

    #return prod(numers) / prod(denoms)
    return exp(sum(log.(numers)) - sum(log.(denoms)))
end

function squarify(ψ::ITensorNetwork)
    ψ = copy(ψ)
    diagonal_edges = filter(e -> e ∈ vcat(edges(ψ), reverse.(edges(ψ))), [NamedEdge(v => v .+ (1,1)) for v in vertices(ψ)])

    for e in diagonal_edges
        l = commonind(ψ[src(e)], ψ[dst(e)])
        sim_l = sim(l)
        ITensorNetworks.@preserve_graph ψ[dst(e)] = ψ[dst(e)] * delta(l, sim_l)
        vm = src(e) .+ (1, 0)
        ITensorNetworks.@preserve_graph ψ[vm] = ψ[vm] * delta(l, sim_l)
    end

    ψ = ITensorNetworks.combine_linkinds(ψ)
    @assert isempty(intersect(diagonal_edges, edges(ψ)))
    
    antidiagonal_edges = filter(e -> e ∈ vcat(edges(ψ), reverse.(edges(ψ))), [NamedEdge(v => v .+ (-1,1)) for v in vertices(ψ)])

    for e in antidiagonal_edges
        l = commonind(ψ[src(e)], ψ[dst(e)])
        sim_l = sim(l)
        ITensorNetworks.@preserve_graph ψ[dst(e)] = ψ[dst(e)] * delta(l, sim_l)
        vm = src(e) .+ (-1, 0)
        ITensorNetworks.@preserve_graph ψ[vm] = ψ[vm] * delta(l, sim_l)
    end

    ψ = ITensorNetworks.combine_linkinds(ψ)
    @assert isempty(intersect(antidiagonal_edges, edges(ψ)))

    return ψ
end

function main(n ,maxdims)  

    CUDA.reclaim()
    
    ITensors.disable_warn_order()
    cutoff = nothing

    

    g = queen_graph(n)
    s = siteinds("S=1/2", g)
    ψ_cpu = build_queenstate(s, n)

    ψ_cpu = squarify(ψ_cpu)
    ψ_cpu = project_state(ψ_cpu)


    z = 1
    # sf = (n *0.146)^(1/n)
    # for v in vertices(ψ_cpu)
    #     ψ_cpu[v] =  (1/sf) *  ψ_cpu[v]
    #     z *= sf
    # end

    T = ITensorMPS.MPO([ψ_cpu[(i,2)] for i in 1:n])
    s = Index[commonind(ψ_cpu[(i,1)], ψ_cpu[(i,2)]) for i in 1:n]

    for i in 1:n
        T[n] = replaceind(T[n], commonind(ψ_cpu[(i,2)], ψ_cpu[(i,3)]), prime(s[i]))
        #T[n] = dag(swapprime(T[n], 0 => 1))
    end

    χ = 50
    #T = ITensorMPS.random_mpo(s, 1)
    Xi = ITensorMPS.normalize(ITensorMPS.randomMPS(s; linkdims = χ))
    ITensorMPS.orthogonalize!(Xi, 1)
    niters=  1000

    Xis = []
    for i in 1:niters
        Xip1 = normalize(ITensorMPS.apply(T, Xi; maxdim =χ, cutoff = 0))
        ITensorMPS.orthogonalize!(Xip1, 1)
        c = conj(inner(Xip1, Xi))
        Xip1 *= conj(c) / abs(c)
        Xi = copy(Xip1)
        push!(Xis, Xi)

        i > 1 && @show inner(Xis[i], Xis[i-1])
    end

    
end

#n = parse(Int64, ARGS[1])
n =4
maxdims = [128]
main(n, maxdims)