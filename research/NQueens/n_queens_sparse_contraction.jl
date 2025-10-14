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

using NPZ

using StatsBase

using Base.Threads

struct sparseinttensor
    indices
    nonzeros
end

function to_bool(a::Int)
    a == 2 && return true
    a == 1 && return false
    error("Can't convert")
end

function convert_to_sparseitensor(t::ITensor)
    inds = ITensors.inds(t)

    eachindval = [i for i in ITensors.eachindval(inds)]
    nonzeros = filter(i -> t[i...] == 1.0, eachindval)
    nonzeros = collect(map(v -> collect(BitArray(to_bool.(last.(v)))), nonzeros))

    return sparseinttensor(Index[i for i in inds], nonzeros)
end

function unique_nonzeros(t::sparseinttensor)
    return length(unique(t.nonzeros))
end

function multiply_sparseitensors(t1::sparseinttensor, t2::sparseinttensor)

    length(t1.nonzeros) > length(t2.nonzeros) && return multiply_sparseitensors(t2,t1)
    t1_inds, t2_inds = t1.indices, t2.indices

    t1_uncommoninds = collect(filter(t1_ind -> t1_ind ∉ t2_inds, t1_inds))
    t2_uncommoninds = collect(filter(t2_ind -> t2_ind ∉ t1_inds, t2_inds))
    uncommoninds = vcat(t1_uncommoninds, t2_uncommoninds)

    t1_commoninds = collect(filter(t1_ind -> t1_ind ∉ uncommoninds, t1_inds))
    t2_commoninds = collect(filter(t2_ind -> t2_ind ∉ uncommoninds, t2_inds))

    perm = indexin(t2_commoninds, t1_commoninds)
    @assert t1_commoninds[perm] == t2_commoninds

    t1_common_ind_pos = filter(i -> t1_inds[i] ∈ t1_commoninds, [i for i in 1:length(t1_inds)])
    t2_common_ind_pos = filter(i -> t2_inds[i] ∈ t2_commoninds, [i for i in 1:length(t2_inds)])
    t1_uncommon_ind_pos = filter(i -> t1_inds[i] ∈ uncommoninds, [i for i in 1:length(t1_inds)])
    t2_uncommon_ind_pos = filter(i -> t2_inds[i] ∈ uncommoninds, [i for i in 1:length(t2_inds)])

    nonzeros = []
    for t1_nonzero in t1.nonzeros
        t1_nonzero_common = t1_nonzero[t1_common_ind_pos][perm]
        t1_nonzero_uncommon = t1_nonzero[t1_uncommon_ind_pos]
        for t2_nonzero in t2.nonzeros

            t2_nonzero_common = t2_nonzero[t2_common_ind_pos]
            t2_nonzero_uncommon = t2_nonzero[t2_uncommon_ind_pos]

            if t1_nonzero_common == t2_nonzero_common
                nonzeros = push!(nonzeros, vcat(t1_nonzero_uncommon, t2_nonzero_uncommon))
            end

        end
    end

    return sparseinttensor(uncommoninds, nonzeros)
end

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

function sparse_multiply_sequence(ψ, vertex_sequence; init_sparsetensor = nothing)

    if init_sparsetensor == nothing
        sparsetensor = convert_to_sparseitensor(ψ[first(vertex_sequence)])
    else
        sparsetensor = multiply_sparseitensors(init_sparsetensor, convert_to_sparseitensor(ψ[first(vertex_sequence)]))
    end

    for v in vertex_sequence[2:length(vertex_sequence)]
        sparsetensor = multiply_sparseitensors(sparsetensor, convert_to_sparseitensor(ψ[v]))
    end
    return sparsetensor
end

function select_nonzeros(c::sparseinttensor, n::Int64 = 1)
    return sparseinttensor(c.indices, StatsBase.sample(c.nonzeros, n; replace = false))
end

function sparse_contract(ψ, n, trim_frequency = n +1)

    no_paths = 1

    c1 = convert_to_sparseitensor(ψ[(1,1)])
    c1 = sparse_multiply_sequence(ψ, [(i, 1) for i in 2:n]; init_sparsetensor = c1)


    println("Boundary Column has $(length(c1.nonzeros)) nonzeros, $(unique_nonzeros(c1)) of which are unique")

    for j in 2:n
        c1 = sparse_multiply_sequence(ψ, [(i, j) for i in 1:n]; init_sparsetensor = c1)
        println("Column $j has $(length(c1.nonzeros)) nonzeros, $(unique_nonzeros(c1)) of which are unique")

        isempty(c1.nonzeros) && return 0

        if j % trim_frequency == 1
            no_paths *= length(c1.nonzeros)
            c1 = select_nonzeros(c1, 1)
        end
    end

    nsols = length(c1.nonzeros)
    return nsols, no_paths
end
    


function main(n)  
    
    ITensors.disable_warn_order()
    g = queen_graph(n)
    s = siteinds("S=1/2", g)
    println("State loaded for n is $(n)")
    ψ = build_queenstate(s, n)
    pre_placed_queens = []
    pre_placed_no_queens = []
    ψ = project_state(ψ; pre_placed_queens, pre_placed_no_queens)

    sparse_contract(ψ, n)


end

n = 27
main(n)  