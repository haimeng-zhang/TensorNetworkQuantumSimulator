using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using NPZ

using ITensorMPS
using Statistics
using EinExprs

using ITensorNetworks: IndsNetwork, AbstractBeliefPropagationCache, BeliefPropagationCache, random_tensornetwork, incoming_messages, edge_tag,
    set_message!

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, inds, commoninds

using NamedGraphs.GraphsExtensions: add_edges, add_vertices, rem_vertices

using NamedGraphs.PartitionedGraphs: PartitionVertex, PartitionEdge, partitionedge

using ITensorNetworks: ITensorsExtensions

function _apply(gate::ITensor, ψ::ITensorNetwork, ψψ::BeliefPropagationCache, vs = ITensorNetworks.neighbor_vertices(ψ, gate); apply_kwargs)
    ψ = copy(ψ)
    s = siteinds(ψ)
    g, _ = induced_subgraph(ITensorNetworks.underlying_graph(ψ), vs)
    vc = only(center(g))
    vls = collect(NamedGraphs.GraphsExtensions.leaf_vertices(g))

    for vl in vls
        ms = incoming_messages(ψψ, PartitionVertex(vl); ignore_edges = [PartitionEdge(vc => vl)])
        for m in ms
            @assert ndims(m) == 2
            sqrt_m = ITensorsExtensions.map_eigvals(sqrt, m, inds(m)[1], inds(m)[2]; cutoff = nothing, ishermitian=true)
            ITensorNetworks.@preserve_graph ψ[vl] = noprime(ψ[vl] * sqrt_m)
        end

        ψvl, R = qr(ψ[vl], uniqueinds(uniqueinds(ψ[vl], ψ[vc]), only(s[vl])))
        ITensorNetworks.@preserve_graph ψ[vl] = ψvl
        ITensorNetworks.@preserve_graph ψ[vc] = ψ[vc] * R
    end

    ITensorNetworks.@preserve_graph ψ[vc] = noprime(ψ[vc] * gate)

    for vl in vls
        linds = unioninds(commoninds(ψ[vl], ψ[vc]), only(s[vl]))
        e = NamedEdge(vc => vl)
        singular_values! = Ref(ITensor())
        Rvl, Rvc, spec = ITensors.factorize_svd(ψ[vc], linds; ortho="none", tags=edge_tag(e), singular_values!, apply_kwargs...)
        ITensorNetworks.@preserve_graph ψ[vl] = noprime(ψ[vl] * Rvl)
        ITensorNetworks.@preserve_graph ψ[vc] = noprime(Rvc)

        S = singular_values![]
        pe = partitionedge(ψψ, (vc, "bra") => (vl, "bra"))
        lind = commonind(S, ψ[vl])
        δuv = dag(copy(S))
        δuv = replaceind(δuv, lind, lind')
        map_diag!(sign, δuv, δuv)
        S = denseblocks(S) * denseblocks(δuv)
        set_message!(ψψ, pe, dag.(ITensor[S]))
        set_message!(ψψ, reverse(pe), ITensor[S])

        ms = incoming_messages(ψψ, PartitionVertex(vl); ignore_edges = [PartitionEdge(vc => vl)])
        for m in ms
            @assert ndims(m) == 2
            inv_sqrt_m = ITensorsExtensions.map_eigvals(inv ∘ sqrt, m, inds(m)[1], inds(m)[2]; cutoff = nothing, ishermitian=true)
            ITensorNetworks.@preserve_graph ψ[vl] = noprime(ψ[vl] * dag(inv_sqrt_m))
        end
    end

    for v in vcat(vls, [vc])
        ITensorNetworks.@preserve_graph ψψ[(v, "ket")] = ψ[v]
        ITensorNetworks.@preserve_graph ψψ[(v, "bra")] = prime(dag(ψ[v]))
    end

    return ψ, ψψ
end

function move_site_indices_to_corner(ψ::ITensorNetwork, corner_vertex)
    ψ = copy(ψ)
    for edge_vertex in neighbors(ψ, corner_vertex)
        ψ = move_site_index_to_corner(ψ, corner_vertex, edge_vertex)
    end
    return ψ
end

function _expect(ψ::ITensor, s, obs::Tuple, edge_vertices)
    i = findfirst(vc -> vc == only(obs[2]), edge_vertices)
    op = ITensors.op(first(obs), s[i])
    ψO = noprime(ψ * op)
    ψOψ = ψO * dag(ψ)
    return ψOψ[]
end

function move_site_indices_to_edge(ψ::ITensorNetwork, corner_vertex, s_old::IndsNetwork)
    ψ = copy(ψ)
    for edge_vertex in neighbors(ψ, corner_vertex)
        se = only(s_old[edge_vertex])
        ψ = move_site_index_to_edge(ψ, corner_vertex, edge_vertex, se)
    end
    return ψ
end

function move_site_index_to_corner(ψ::ITensorNetwork, corner_vertex, edge_vertex)
    ψ = copy(ψ)
    A, B = ψ[edge_vertex], ψ[corner_vertex]
    s = only(ITN.siteinds(ψ, edge_vertex))
    linds = setdiff(uniqueinds(A, B), [s])
    U, V = ITensors.factorize_svd(A, linds; cutoff = 1e-16, tags = tags(commonind(A, B)))
    ψ[edge_vertex] = U
    ψ[corner_vertex] = V * B
    return ψ
end

function move_site_index_to_edge(ψ::ITensorNetwork, corner_vertex, edge_vertex, s::Index)
    ψ = copy(ψ)
    A, B = ψ[edge_vertex], ψ[corner_vertex]
    linds = setdiff(uniqueinds(B, A), [s])
    U, V = ITensors.factorize_svd(B, linds; cutoff = 1e-16, tags = tags(commonind(A, B)))
    ψ[corner_vertex] = U
    ψ[edge_vertex] = V * A
    return ψ
end

function apply_corner_gate(ψ::ITensorNetwork, s::IndsNetwork, corner_vertex, gate::ITensor)
    ψ = copy(ψ)
    ψ = move_site_indices_to_corner(ψ, corner_vertex)
    ψ[corner_vertex] = noprime(ψ[corner_vertex] * gate)
    ψ = move_site_indices_to_edge(ψ, corner_vertex, s)
    return ψ
end

function build_case_a_graph()
    return named_grid((4,4); periodic = true)
end

function build_case_b_graph()
   g = named_grid((4,5))
   g = rem_vertices(g, [(1,1), (4,1), (1,5), (4,5)])
   return g
end

function build_case_c_graph()
    g = NamedGraphs.NamedGraphGenerators.named_hexagonal_lattice_graph(2,3)
    g = add_vertices(g, [(0, 2), (0, 3)])
    g = add_edges(g, [NamedEdge((0, 2) => (0, 3)), NamedEdge((1, 2) => (0, 2)), NamedEdge((1, 3) => (0, 3))])
    return g
end

function build_parity_check_gate(sinds::Vector{<:Index}, a::Number, b::Number, c::Number, d::Number)

    U = reduce(*, [ITensors.op("Id", s) for s in sinds])
    U = U + (a-1) * reduce(*, [ITensors.op("ProjUp", s) for s in sinds])
    U = U + (d-1) * reduce(*, [ITensors.op("ProjDn", s) for s in sinds])
    U = U + b * reduce(*, [ITensors.op("S-", s) for s in sinds])
    U = U + c * reduce(*, [ITensors.op("S+", s) for s in sinds])

    return U
end

function main_hexagonal(θ::Float64, ϕ::Float64, nlayers::Int64, maxdim::Int64)

    η, ω = 0.0, 0.0
    a, b, c, d = cos(θ) * exp( im * (ϕ + η)), sin(θ) * exp( im * (ϕ + ω)), -sin(θ) * exp( im * (ϕ - ω)),  cos(θ) * exp( im * (ϕ - η))
    U = [a b; c d]
    @assert norm(U * conj(transpose(U)) - [1 0; 0 1]) <= 1e-10

    g = NamedGraphs.NamedGraphGenerators.named_hexagonal_lattice_graph(7,7)

    egs = NamedGraphs.edgeinduced_subgraphs_no_leaves(g, 6)
    println("Number of primitive loops is $(length(egs))")
    corner_vertices = collect(vertices(g))
    g = NamedGraphs.GraphsExtensions.decorate_graph_edges(g)
    edge_vertices = setdiff(collect(vertices(g)), corner_vertices)

    s = ITensorNetworks.siteinds("S=1/2", g)
    ψ0 = ITensorNetwork(v -> "Z+", s)
    for v in corner_vertices
        ψ0[v] = ψ0[v] * onehot(only(s[v]) => 1)
    end

    ψ = copy(ψ0)
    s = ITensorNetworks.siteinds(ψ)

    ψψ = build_bp_cache(ψ)

    A_sublattice, B_sublattice = filter(v -> isodd(sum(v)), corner_vertices), filter(v -> iseven(sum(v)), corner_vertices)
    M_obs = [("Z", [v]) for v in edge_vertices]
    zs = ComplexF64[sum(TN.expect(ψψ, M_obs))]
    fs = [1.0]

    apply_kwargs = (; maxdim = maxdim, cutoff = 1e-12)

    for i in 1:nlayers
        println("On layer $i")

        @assert issetequal(vcat(A_sublattice, B_sublattice), corner_vertices)
        for sublattice in [A_sublattice, B_sublattice]
            for v in sublattice
                sinds = [only(s[vn]) for vn in neighbors(g, v)]
                t = build_parity_check_gate(sinds, a, b, c, d)
                vs = vcat(ITensorNetworks.neighbor_vertices(ψ, t), [v])
                ψ, ψψ = _apply(t, ψ, ψψ, vs; apply_kwargs)
            end

            ψψ = updatecache(ψψ)
        end

        f  = inner(ψ, ψ0; alg = "bp")
        F = f * conj(f)
        _zs = TN.expect(ψψ, M_obs)
        push!(zs, sum(_zs))
        push!(fs, F)

        @show ITensorNetworks.maxlinkdim(ψ)
    end

    return real.(zs), real.(fs)
end

function main(graph_case::String, parameter_case::Int64, nlayers::Int64)

    if parameter_case == 1
        θ, ϕ, η, ω = 1.0, 0.9, 0.6, 0.4
    elseif parameter_case == 2
        θ, ϕ, η, ω = 0.3, 0.5, 0.7, 0.9
    elseif parameter_case == 3
        θ, ϕ, η, ω = 1.1, 0, 0, 0
    end
    a, b, c, d = cos(θ) * exp( im * (ϕ + η)), sin(θ) * exp( im * (ϕ + ω)), -sin(θ) * exp( im * (ϕ - ω)),  cos(θ) * exp( im * (ϕ - η))
    U = [a b; c d]
    @assert norm(U * conj(transpose(U)) - [1 0; 0 1]) <= 1e-10

    g = graph_case == "A" ? build_case_a_graph() : graph_case == "B" ? build_case_b_graph() : build_case_c_graph()
    corner_vertices = collect(vertices(g))
    g = NamedGraphs.GraphsExtensions.decorate_graph_edges(g)
    edge_vertices = setdiff(collect(vertices(g)), corner_vertices)
    @show length(edge_vertices)


    s = ITensorNetworks.siteinds("S=1/2", g)
    ψ0 = ITensorNetwork(v -> "Z+", s)
    for v in corner_vertices
        ψ0[v] = ψ0[v] * onehot(only(s[v]) => 1)
    end

    ψ = copy(ψ0)
    s = ITensorNetworks.siteinds(ψ)

    ψψ = build_bp_cache(ψ)

    maxdim = 10

    A_sublattice, B_sublattice = filter(v -> isodd(sum(v)), corner_vertices), filter(v -> iseven(sum(v)), corner_vertices)
    M_obs = [("Z", [v]) for v in edge_vertices]
    zs = ComplexF64[sum(TN.expect(ψψ, M_obs))]
    fs = [1.0]

    apply_kwargs = (; maxdim = maxdim, cutoff = 1e-12)

    for i in 1:nlayers
        println("On layer $i")

        @assert issetequal(vcat(A_sublattice, B_sublattice), corner_vertices)
        for sublattice in [A_sublattice, B_sublattice]
            for v in sublattice
                sinds = [only(s[vn]) for vn in neighbors(g, v)]
                t = build_parity_check_gate(sinds, a, b, c, d)
                #ψ = apply_corner_gate(ψ, s, v, t)
                vs = vcat(ITensorNetworks.neighbor_vertices(ψ, t), [v])
                ψ, ψψ = _apply(t, ψ, ψψ, vs; apply_kwargs)
            end

            ψψ = updatecache(ψψ)
        end

        #ψψ = build_bp_cache(ψ)
        #ψ = truncate(ψ; cache! = Ref(ψψ), update_cache = false, maxdim = maxdim)
        #ψψ = build_bp_cache(ψ)
        f  = inner(ψ, ψ0; alg = "bp")
        F = f * conj(f)
        _zs = TN.expect(ψψ, M_obs)
        push!(zs, sum(_zs))
        push!(fs, F)

        @show ITensorNetworks.maxlinkdim(ψ)
    end

    @show vcat([0], [i for i in 1:nlayers])
    @show real.(zs)
    @show real.(fs)

end

function main_mps(graph_case::String, parameter_case::Int64, nlayers::Int64)

    ITensors.disable_warn_order()

    if parameter_case == 1
        θ, ϕ, η, ω = 1.0, 0.9, 0.6, 0.4
    elseif parameter_case == 2
        θ, ϕ, η, ω = 0.3, 0.5, 0.7, 0.9
    elseif parameter_case == 3
        θ, ϕ, η, ω = 1.1, 0, 0, 0
    end
    a, b, c, d = cos(θ) * exp( im * (ϕ + η)), sin(θ) * exp( im * (ϕ + ω)), -sin(θ) * exp( im * (ϕ - ω)),  cos(θ) * exp( im * (ϕ - η))

    g = graph_case == "A" ? build_case_a_graph() : graph_case == "B" ? build_case_b_graph() : build_case_c_graph()

    corner_vertices = collect(vertices(g))
    g = NamedGraphs.GraphsExtensions.decorate_graph_edges(g)
    edge_vertices = setdiff(collect(vertices(g)), corner_vertices)

    @show length(edge_vertices)

    s = ITensorMPS.siteinds("S=1/2", length(edge_vertices))
    ψ0 = reduce(*, [onehot(s[i] => 1) for i in 1:length(s)])
    ψ = copy(ψ0)

    A_sublattice, B_sublattice = filter(v -> isodd(sum(v)), corner_vertices), filter(v -> iseven(sum(v)), corner_vertices)
    M_obs = [("Z", [v]) for v in edge_vertices]
    zs = ComplexF64[sum([_expect(ψ, s, obs, edge_vertices) for obs in M_obs])]
    Fs = [1.0]
    for i in 1:nlayers
        println("On layer $i")

        for sublattice in [A_sublattice, B_sublattice]
            for v in sublattice
                is = [findfirst(vc -> vc == vn, edge_vertices) for vn in neighbors(g, v)]
                sinds = [s[i] for i in is]
                t = build_parity_check_gate(sinds, a, b, c, d)
                ψ = noprime(ψ * t)
            end
        end

        _zs =[_expect(ψ, s, obs, edge_vertices) for obs in M_obs]
        f = (ψ * ψ0)[]
        push!(Fs, f * conj(f))
        push!(zs, sum(_zs))
    end

    @show zs
    @show Fs

end
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using NPZ

using ITensorMPS
using Statistics
using EinExprs

using ITensorNetworks: IndsNetwork, AbstractBeliefPropagationCache, BeliefPropagationCache, random_tensornetwork, incoming_messages, edge_tag,
    set_message!

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, inds, commoninds

using NamedGraphs.GraphsExtensions: add_edges, add_vertices, rem_vertices

using NamedGraphs.PartitionedGraphs: PartitionVertex, PartitionEdge, partitionedge

using ITensorNetworks: ITensorsExtensions

function _apply(gate::ITensor, ψ::ITensorNetwork, ψψ::BeliefPropagationCache, vs = ITensorNetworks.neighbor_vertices(ψ, gate); apply_kwargs, regularization = 1e-14)
    ψ = copy(ψ)
    ψψ = copy(ψψ)
    s = ITensorNetworks.siteinds(ψ)
    g, _ = induced_subgraph(ITensorNetworks.underlying_graph(ψ), vs)
    vc = only(center(g))
    vls = collect(NamedGraphs.GraphsExtensions.leaf_vertices(g))

    for vl in vls
        ms = incoming_messages(ψψ, PartitionVertex(vl); ignore_edges = [PartitionEdge(vc => vl)])
        for m in ms
            @assert ndims(m) == 2
            sqrt_m = ITensorsExtensions.map_eigvals(x -> sqrt(x + regularization), m, inds(m)[1], inds(m)[2]; cutoff = nothing, ishermitian=true)
            ITensorNetworks.@preserve_graph ψ[vl] = noprime(ψ[vl] * sqrt_m)
        end

        ψvl, R = qr(ψ[vl], uniqueinds(uniqueinds(ψ[vl], ψ[vc]), only(s[vl])))
        ITensorNetworks.@preserve_graph ψ[vl] = ψvl
        ITensorNetworks.@preserve_graph ψ[vc] = ψ[vc] * R
    end

    ITensorNetworks.@preserve_graph ψ[vc] = noprime(ψ[vc] * gate)

    for vl in vls
        linds = unioninds(commoninds(ψ[vl], ψ[vc]), only(s[vl]))
        e = NamedEdge(vc => vl)
        singular_values! = Ref(ITensor())
        Rvl, Rvc, spec = ITensors.factorize_svd(ψ[vc], linds; ortho="none", tags=edge_tag(e), singular_values!, apply_kwargs...)
        ITensorNetworks.@preserve_graph ψ[vl] = noprime(ψ[vl] * Rvl)
        ITensorNetworks.@preserve_graph ψ[vc] = normalize(noprime(Rvc))

        S = singular_values![]
        pe = partitionedge(ψψ, (vc, "bra") => (vl, "bra"))
        lind = commonind(S, ψ[vl])
        δuv = dag(copy(S))
        δuv = replaceind(δuv, lind, lind')
        map_diag!(sign, δuv, δuv)
        S = denseblocks(S) * denseblocks(δuv)
        set_message!(ψψ, pe, dag.(ITensor[S]))
        set_message!(ψψ, reverse(pe), ITensor[S])

        ms = incoming_messages(ψψ, PartitionVertex(vl); ignore_edges = [PartitionEdge(vc => vl)])
        for m in ms
            @assert ndims(m) == 2
            inv_sqrt_m = ITensorsExtensions.map_eigvals(x -> (inv ∘ sqrt)(x + regularization), m, inds(m)[1], inds(m)[2]; cutoff = nothing, ishermitian=true)
            ITensorNetworks.@preserve_graph ψ[vl] = normalize(noprime(ψ[vl] * dag(inv_sqrt_m)))
        end
    end

    for v in vcat(vls, [vc])
        ITensorNetworks.@preserve_graph ψψ[(v, "ket")] = ψ[v]
        ITensorNetworks.@preserve_graph ψψ[(v, "bra")] = prime(dag(ψ[v]))
    end

    return ψ, ψψ
end

function move_site_indices_to_corner(ψ::ITensorNetwork, corner_vertex)
    ψ = copy(ψ)
    for edge_vertex in neighbors(ψ, corner_vertex)
        ψ = move_site_index_to_corner(ψ, corner_vertex, edge_vertex)
    end
    return ψ
end

function _expect(ψ::ITensor, s, obs::Tuple, edge_vertices)
    i = findfirst(vc -> vc == only(obs[2]), edge_vertices)
    op = ITensors.op(first(obs), s[i])
    ψO = noprime(ψ * op)
    ψOψ = ψO * dag(ψ)
    return ψOψ[]
end

function move_site_indices_to_edge(ψ::ITensorNetwork, corner_vertex, s_old::IndsNetwork)
    ψ = copy(ψ)
    for edge_vertex in neighbors(ψ, corner_vertex)
        se = only(s_old[edge_vertex])
        ψ = move_site_index_to_edge(ψ, corner_vertex, edge_vertex, se)
    end
    return ψ
end

function move_site_index_to_corner(ψ::ITensorNetwork, corner_vertex, edge_vertex)
    ψ = copy(ψ)
    A, B = ψ[edge_vertex], ψ[corner_vertex]
    s = only(ITN.siteinds(ψ, edge_vertex))
    linds = setdiff(uniqueinds(A, B), [s])
    U, V = ITensors.factorize_svd(A, linds; cutoff = 1e-16, tags = tags(commonind(A, B)))
    ψ[edge_vertex] = U
    ψ[corner_vertex] = V * B
    return ψ
end

function move_site_index_to_edge(ψ::ITensorNetwork, corner_vertex, edge_vertex, s::Index)
    ψ = copy(ψ)
    A, B = ψ[edge_vertex], ψ[corner_vertex]
    linds = setdiff(uniqueinds(B, A), [s])
    U, V = ITensors.factorize_svd(B, linds; cutoff = 1e-16, tags = tags(commonind(A, B)))
    ψ[corner_vertex] = U
    ψ[edge_vertex] = V * A
    return ψ
end

function apply_corner_gate(ψ::ITensorNetwork, s::IndsNetwork, corner_vertex, gate::ITensor)
    ψ = copy(ψ)
    ψ = move_site_indices_to_corner(ψ, corner_vertex)
    ψ[corner_vertex] = noprime(ψ[corner_vertex] * gate)
    ψ = move_site_indices_to_edge(ψ, corner_vertex, s)
    return ψ
end

function build_case_a_graph()
    return named_grid((4,4); periodic = true)
end

function build_case_b_graph()
   g = named_grid((4,5))
   g = rem_vertices(g, [(1,1), (4,1), (1,5), (4,5)])
   return g
end

function build_case_c_graph()
    g = NamedGraphs.NamedGraphGenerators.named_hexagonal_lattice_graph(2,3)
    g = add_vertices(g, [(0, 2), (0, 3)])
    g = add_edges(g, [NamedEdge((0, 2) => (0, 3)), NamedEdge((1, 2) => (0, 2)), NamedEdge((1, 3) => (0, 3))])
    return g
end

function build_parity_check_gate(sinds::Vector{<:Index}, a::Number, b::Number, c::Number, d::Number)

    U = reduce(*, [ITensors.op("Id", s) for s in sinds])
    U = U + (a-1) * reduce(*, [ITensors.op("ProjUp", s) for s in sinds])
    U = U + (d-1) * reduce(*, [ITensors.op("ProjDn", s) for s in sinds])
    U = U + b * reduce(*, [ITensors.op("S-", s) for s in sinds])
    U = U + c * reduce(*, [ITensors.op("S+", s) for s in sinds])

    return U
end

function main_hexagonal(θ::Float64, ϕ::Float64, nlayers::Int64, maxdim::Int64)

    η, ω = 0.0, 0.0
    a, b, c, d = cos(θ) * exp( im * (ϕ + η)), sin(θ) * exp( im * (ϕ + ω)), -sin(θ) * exp( im * (ϕ - ω)),  cos(θ) * exp( im * (ϕ - η))
    U = [a b; c d]
    @assert norm(U * conj(transpose(U)) - [1 0; 0 1]) <= 1e-10

    g = NamedGraphs.NamedGraphGenerators.named_hexagonal_lattice_graph(7,7)

    egs = NamedGraphs.edgeinduced_subgraphs_no_leaves(g, 6)
    println("Number of primitive loops is $(length(egs))")
    flush(stdout)
    corner_vertices = collect(vertices(g))
    g = NamedGraphs.GraphsExtensions.decorate_graph_edges(g)
    edge_vertices = setdiff(collect(vertices(g)), corner_vertices)

    s = ITensorNetworks.siteinds("S=1/2", g)
    ψ0 = ITensorNetwork(v -> "Z+", s)
    for v in corner_vertices
        ψ0[v] = ψ0[v] * onehot(only(s[v]) => 1)
    end

    ψ = copy(ψ0)
    s = ITensorNetworks.siteinds(ψ)

    ψψ = build_bp_cache(ψ)

    A_sublattice, B_sublattice = filter(v -> isodd(sum(v)), corner_vertices), filter(v -> iseven(sum(v)), corner_vertices)
    M_obs = [("Z", [v]) for v in edge_vertices]
    zs = ComplexF64[sum(TN.expect(ψψ, M_obs))]
    fs = [1.0]

    apply_kwargs = (; maxdim = maxdim, cutoff = 1e-12)

    for i in 1:nlayers
        println("On layer $i")

        @assert issetequal(vcat(A_sublattice, B_sublattice), corner_vertices)
        for sublattice in [A_sublattice, B_sublattice]
            for v in sublattice
                sinds = [only(s[vn]) for vn in neighbors(g, v)]
                t = build_parity_check_gate(sinds, a, b, c, d)
                vs = vcat(ITensorNetworks.neighbor_vertices(ψ, t), [v])
                ψ, ψψ = _apply(t, ψ, ψψ, vs; apply_kwargs)
            end

            ψψ = updatecache(ψψ; message_update_kwargs=(; message_update_function=TN.default_posdef_message_update_function))
        end

        f  = inner(ψ, ψ0; alg = "bp")
        F = f * conj(f)
        _zs = TN.expect(ψψ, M_obs)
        push!(zs, sum(_zs))
        push!(fs, F)

        @show ITensorNetworks.maxlinkdim(ψ)
        flush(stdout)
    end

    return real.(zs), real.(fs)
end

θ, ϕ, maxdim, nlayers = parse(Float64, ARGS[1]), parse(Float64, ARGS[2]), parse(Int64, ARGS[3]), parse(Int64, ARGS[4])
#θ, ϕ, nlayers, maxdim = 1.1, 0.5, 5, 16
ZBPs, Fs = main_hexagonal(θ, ϕ, nlayers, maxdim)

file_name =  "/mnt/home/jtindall/ceph/Data/PCCircuits/Hexagonal/Theta"*string(θ) *"Phi"*string(ϕ) * "Nlayers" * string(nlayers) * "maxdim" * string(maxdim) * ".npz"
npzwrite(file_name, Fs =Fs, ZBPs = ZBPs)
