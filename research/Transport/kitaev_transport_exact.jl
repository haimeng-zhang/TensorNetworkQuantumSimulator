using LinearAlgebra, SparseArrays

using NamedGraphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: vertices, add_edge!, src, dst, rem_edge!, degree, neighbors
using Dictionaries
using Statistics

using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks: siteinds, ITensorNetwork, ITensorNetworks, expect
const ITN = ITensorNetworks

using Random

using NPZ

function edge_to_bond(e::NamedEdge, vs::Vector)
    return findfirst(v -> v == src(e), vs), findfirst(v -> v == dst(e), vs)
end

function half_periodic_hexagonal_lattice(nx::Int64, ny::Int64)
    g = named_grid((ny, nx))

    nx == 2 && return g

    for i in 1:ny
        if isodd(i)
            g = add_edge!(g, (i, 1) => (i, nx))
        end
    end

    for e in edges(g)
        v1, v2 = src(e), dst(e)
        r1, r2 = first(v1), first(v2)
        if r1 == r2
            column = last(v1) < last(v2) ? last(v1) : last(v2)
            column = column == 1 && last(v2) == nx ? nx : column
            if iseven(column + r1) 
                g = rem_edge!(g, e)
            end
        end
    end
    return g
end

function build_flux_free_adj_mat(g)
    nv = length(vertices(g))
    Adj_mat = zeros((nv,nv))
    vs = collect(vertices(g))

    for v in vertices(g)
        for vn in neighbors(g, v)
            (i, j) = edge_to_bond(NamedEdge(v => vn), vs)
            Adj_mat[i, j] = i > j ? - 2 : 2
            Adj_mat[j, i] = i > j ? 2 : - 2
        end
    end
    return Adj_mat
end

function retain_edges_adj_mat(A, g, es_to_retain)
    A = copy(A)

    nv = length(vertices(g))
    vs = collect(vertices(g))
    ec = edge_color(g, 3)

    for e in edges(g)
        if e ∉ es_to_retain && reverse(e) ∉ es_to_retain
            (src_v, dst_v) = edge_to_bond(e, vs)
            A[src_v, dst_v] = 0
            A[dst_v, src_v] = 0
        end
    end

    return A
end

function edge_energy(g, A, Γ, e)
    (i,j) = edge_to_bond(e, collect(vertices(g)))
    return -0.5 * im * A[i, j] * Γ[i,j]
end

function energy(g, A, Γ, es::Vector)
    return sum([edge_energy(g, A, Γ, e) for e in es])
end

function total_energy(A, Γ)
    return -0.25 * im * tr(A * Γ)
end

function randomly_sample_plaquettes(A, g)
    A_sampled = copy(A)

    egs=  NamedGraphs.edgeinduced_subgraphs_no_leaves(g, 6)
    egs = filter(eg -> length(edges(eg)) == 6, egs)

    nsteps = 10000
    for i in 1:nsteps
        eg = rand(egs)
        if rand() <= 0.5
            es = edges(eg)
            for e in es
                (i,j) = edge_to_bond(e, collect(vertices(g)))
                A_sampled[i,j] *= -1
                A_sampled[j,i] *= -1
            end
        end
    end

    W = 1
    for eg in egs
        _W = 1
        for e in edges(eg)
            (i,j) = edge_to_bond(e, collect(vertices(g)))
            if i > j
                _W *= (A_sampled[i,j]/2)
            else
                _W *= (A_sampled[j,i]/2)
            end
        end

        if _W == -1
            W += 1
        end
    end
    return A_sampled, W / length(egs)
end


function main(i::Int64, nx, ny)

    g = half_periodic_hexagonal_lattice(nx,ny)

    nsteps = 300
    dt= 0.1
    ts = [dt*(i-1) for i in 1:(nsteps+1)]
    ϵ = 0.001
    top_vs, bottom_vs = filter(v -> first(v) > ny/2, collect(vertices(g))), filter(v -> first(v) <= ny/2, collect(vertices(g)))

    top_edges = filter(e -> src(e) ∈ top_vs && dst(e) ∈ top_vs, edges(g))
    bottom_edges = filter(e -> src(e) ∈ bottom_vs && dst(e) ∈ bottom_vs, edges(g))
    crossing_edges = setdiff(edges(g), [top_edges; bottom_edges])

    @assert length(top_vs) == length(bottom_vs)
    @assert length(top_edges) == length(bottom_edges)

    top_energies, total_energies = zeros(ComplexF64, (nsteps + 1)), zeros(ComplexF64, (nsteps + 1))

    Random.seed!(1234 * i)
    A = build_flux_free_adj_mat(g)

    A, flux_density = randomly_sample_plaquettes(A, g)

    vals = eigvals(im * A)
    λs = sort(abs.(vals))
    e_gs = - sum(λs) * 0.25 / length(vertices(g))
    println("Ground State e.d is $(e_gs)")

    top_A, bottom_A, cross_A = retain_edges_adj_mat(A, g, top_edges), retain_edges_adj_mat(A, g, bottom_edges), retain_edges_adj_mat(A, g, crossing_edges)

    Γ0 = tanh((ϵ/2) * im * top_A - (ϵ/2) * im * bottom_A)

    e_inits = Statistics.mean([edge_energy(g, A, Γ0, e) for e in top_edges])

    println("Avergae bond energy in top half is $(e_inits)")

    E_top_init, E_bottom_init, E_cross_init  = total_energy(top_A, Γ0), total_energy(bottom_A, Γ0), total_energy(cross_A, Γ0)
    E_top_init = E_top_init + 0.5*E_cross_init
    E_bottom_init = E_bottom_init + 0.5*E_cross_init

    U = exp(0.5 * A * dt)
    Udag = transpose(U)
    Γt = copy(Γ0)
    total_energies[1] += E_top_init + E_bottom_init + E_cross_init

    for j in 1:nsteps
        Γt = U * Γt * Udag
        top_energies[j + 1] += total_energy(top_A, Γt) + 0.5 * total_energy(cross_A, Γt) - E_top_init
        total_energies[j + 1] += (total_energy(bottom_A, Γt) + total_energy(top_A, Γt) + total_energy(cross_A, Γt))
    end

    @show last(top_energies) / nx

    return ts, top_energies, total_energies, flux_density


end

nx, ny, i = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3])
#nx, ny, i = 20, 20, 8
ts, top_energies, total_energies, flux_density = main(i, nx, ny)

file_name =   "/mnt/home/jtindall/ceph/Data/Transport/KitaevModel/ExactRandomFluxSector/ny"*string(ny)*"nx"*string(nx) * "disorderno" * string(i) *".npz"
npzwrite(file_name, ts = ts, top_energies = top_energies, total_energies = total_energies, flux_density = flux_density)