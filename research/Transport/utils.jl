function named_square_cylinder(ny::Int64)
    g = named_grid((ny, 3))
    for i in 1:ny
        g = NG.GraphsExtensions.add_edge(g, NamedEdge((i, 1) => (i, 3)))
    end

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:3])

    bottom_half_vertices = filter(v -> first(v) <= column_lengths[last(v)] / 2, collect(vertices(g)))

    return g, bottom_half_vertices
end

function named_chain(ny::Int64)
    g = named_grid((ny,  1))
    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:1])

    bottom_half_vertices = filter(v -> first(v) <= column_lengths[last(v)] / 2, collect(vertices(g)))

    return g, bottom_half_vertices
end

function named_hexagonal_cylinder(ny::Int64)
    g = named_hexagonal_lattice_graph(ny, 3; periodic = false)

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])

    g = NG.GraphsExtensions.add_vertex(g, (column_lengths[1] + 1, 1))
    g = NG.GraphsExtensions.add_vertex(g, (column_lengths[4] + 1, 4))
    g = NG.GraphsExtensions.add_edge(g, NamedEdge((column_lengths[1], 1) => (column_lengths[1] + 1, 1)))
    g = NG.GraphsExtensions.add_edge(g, NamedEdge((column_lengths[4], 4) => (column_lengths[4] + 1, 4)))

    column_1_vertices=  filter(v -> last(v) == 1, collect(vertices(g)))

    for v in column_1_vertices
       if isempty(filter(v -> last(v) == 2, neighbors(g,v)))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge(v => (first(v), 4)))
       end
    end

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])

    bottom_half_vertices = filter(v -> first(v) <= column_lengths[last(v)] / 2, collect(vertices(g)))

    return g, bottom_half_vertices
end

function named_heavy_hexagonal_cylinder(ny::Int64)
    g1, g2 = named_grid((ny, 1)), NG.GraphsExtensions.rename_vertices(v -> (first(v), 3), named_grid((ny, 1)))
    g = union(g1, g2)

    for i in 1:ny
        if (i-1) % 4 == 0
            g = NG.GraphsExtensions.add_vertex(g, (i, 2))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge((i,1) => (i,2)))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge((i,2) => (i,3)))
        elseif (i -3) % 4 == 0
            g = NG.GraphsExtensions.add_vertex(g, (i, 4))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge((i,3) => (i,4)))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge((i,4) => (i,1)))
        end
    end

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])
    column_lengths[2] = column_lengths[1]
    column_lengths[4] = column_lengths[3]

    bottom_half_vertices = filter(v -> first(v) <= column_lengths[last(v)] / 2, collect(vertices(g)))

    return g, bottom_half_vertices
end

function transport_graph_constructor(g_str::String, ny::Int64)
    g_str == "Hexagonal" && return named_hexagonal_cylinder(ny)
    g_str == "Square" && return named_square_cylinder(ny)
    g_str == "HeavyHexagonal" && return named_heavy_hexagonal_cylinder(ny)
    g_str == "Chain" && return named_chain(ny)
end

function mean_gate_fidelity(errs)
    return prod(1.0 .- errs)^(1.0/length(errs))
end

function honeycomb_kitaev_layer(K::Float64, δt::Float64, ec)
    layer = []
    append!(layer, ("Rxx", e, K*δt) for e in ec[1])
    append!(layer, ("Ryy", e, K*δt) for e in ec[2])
    append!(layer, ("Rzz", e, 2*K*δt) for e in ec[3])
    append!(layer, ("Ryy", e, K*δt) for e in ec[2])
    append!(layer, ("Rxx", e, K*δt) for e in ec[1])
    return layer
end

function honeycomb_kitaev_layers(K::Float64, δt::Float64, ec)
    layers = [[("Rxx", e, K*δt) for e in ec[1]], [("Ryy", e, K*δt) for e in ec[2]], [("Rzz", e, 2*K*δt) for e in ec[3]], [("Ryy", e, K*δt) for e in ec[2]], [("Rxx", e, K*δt) for e in ec[1]]]
    return layers
end

function honeycomb_kitaev_layer_fourth_order(K::Float64, δt::Float64, ec)
    w1, w2 = (1 / (2 - (2^(1/3)))), -((2^(1/3)) / (2 - (2^(1/3))))
    layer = []
    for w in [w1, w2, w1]
        append!(layer, ("Rxx", e, w * K*δt) for e in ec[1])
        append!(layer, ("Ryy", e, w * K*δt) for e in ec[2])
        append!(layer, ("Rzz", e, 2*w * K*δt) for e in ec[3])
        append!(layer, ("Ryy", e, w * K*δt) for e in ec[2])
        append!(layer, ("Rxx", e, w * K*δt) for e in ec[1])
    end

    return layer
end

function honeycomb_kitaev_layer(K::Float64, δt::Float64, ec, vertex_subset)
    layer = []
    append!(layer, ("Rxx", e, K*δt) for e in filter(e -> src(e) ∈ vertex_subset && dst(e) ∈ vertex_subset, ec[1]))
    append!(layer, ("Ryy", e, K*δt) for e in filter(e -> src(e) ∈ vertex_subset && dst(e) ∈ vertex_subset, ec[2]))
    append!(layer, ("Rzz", e, 2*K*δt) for e in filter(e -> src(e) ∈ vertex_subset && dst(e) ∈ vertex_subset, ec[3]))
    append!(layer, ("Ryy", e, K*δt) for e in filter(e -> src(e) ∈ vertex_subset && dst(e) ∈ vertex_subset, ec[2]))
    append!(layer, ("Rxx", e, K*δt) for e in filter(e -> src(e) ∈ vertex_subset && dst(e) ∈ vertex_subset, ec[1]))
    return layer
end

function honeycomb_kitaev_observables(K::Float64, ec, vertex_subset)
    xx_observables = [("XX", pair, K) for pair in ec[1]]
    yy_observables = [("YY", pair, K) for pair in ec[2]]
    zz_observables = [("ZZ", pair, K) for pair in ec[3]]

    obs = [xx_observables; yy_observables; zz_observables]
    obs = filter(obs -> (src(obs[2]) ∈ vertex_subset && dst(obs[2]) ∈ vertex_subset), obs)
    return obs
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