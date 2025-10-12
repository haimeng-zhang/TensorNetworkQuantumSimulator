module TensorNetworkQuantumSimulator


include("imports.jl")
#include("Backend/loopcorrection.jl")
#include("Backend/boundarymps.jl")


include("tensornetworkstate.jl")
include("Backend/abstractbeliefpropagationcache.jl")
include("Backend/beliefpropagationcache.jl")
include("Backend/boundarympscache.jl")
include("graph_ops.jl")
include("utils.jl")
include("constructors.jl")
include("gates.jl")
include("apply.jl")
include("expect.jl")
include("norm_sqr.jl")
include("normalize.jl")
include("Backend/loopcorrection.jl")
include("sampling.jl")
include("symmetric_gauge.jl")


export
    vertices,
    edges,
    apply_gates,
    truncate,
    expect,
    expect_boundarymps,
    expect_loopcorrect,
    fidelity,
    fidelity_boundarymps,
    fidelity_loopcorrect,
    make_hermitian,
    ket_network,
    truncate,
    maxlinkdim,
    siteinds,
    edge_color,
    zerostate,
    getnqubits,
    named_grid,
    sample,
    TensorNetworkState,
    tensornetworkstate,
    random_tensornetworkstate,
    BeliefPropagationCache,
    rescale!,
    message,
    network,
    update,
    symmetric_gauge,
    symmetric_gauge!,
    messages,
    gauge_and_scale
end
