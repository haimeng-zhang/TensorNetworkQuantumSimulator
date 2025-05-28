function named_cylinder(nx::Int64, ny::Int64)
    g = named_grid((nx, ny))
    for i in 1:ny
        g = NG.GraphsExtensions.add_edge(g, NamedEdge((1, i) => (nx, i)))
    end
    return g
end

function named_hexagonal_cylinder(ny::Int64)
    g = named_hexagonal_lattice_graph(ny, 3; periodic = false)
    column_1_vertices=  filter(v -> last(v) == 1, collect(vertices(g)))
    for v in column_1_vertices
       if isempty(filter(v -> last(v) == 2, neighbors(g,v)))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge(v => (first(v), 4)))
       end
    end
    return g
end

function mean_gate_fidelity(errs)
    return prod(1.0 .- errs)^(1.0/length(errs))
end