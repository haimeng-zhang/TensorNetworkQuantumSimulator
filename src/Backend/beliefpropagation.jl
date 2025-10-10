"""
    symmetric_gauge(ψ::AbstractITensorNetwork; cache_update_kwargs = default_posdef_bp_update_kwargs(), kwargs...)

Transform a tensor netework into the symmetric gauge, where the BP message tensors are all diagonal
"""
# function symmetric_gauge(ψ::AbstractITensorNetwork; cache_update_kwargs=default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)), kwargs...)
#     ψ_vidal = VidalITensorNetwork(ψ; cache_update_kwargs, kwargs...)
#     cache_ref = Ref{BeliefPropagationCache}()
#     ψ_symm = ITensorNetwork(ψ_vidal; (cache!)=cache_ref)
#     bp_cache = cache_ref[]
#     return ψ_symm, bp_cache
# end


"""
    entanglement(ψ::ITensorNetwork, e::NamedEdge; (cache!) = nothing, cache_update_kwargs = default_posdef_bp_update_kwargs())

Bipartite Von-Neumann entanglement entropy, estimated, via BP, using the spectrum of the bond tensor on the given edge.
"""
# function entanglement(
#     ψ::ITensorNetwork,
#     e::NamedEdge;
#     (cache!)=nothing,
#     cache_update_kwargs=default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
# )
#     cache = isnothing(cache!) ? build_normsqr_bp_cache(ψ; cache_update_kwargs) : cache![]
#     ψ_vidal = VidalITensorNetwork(ψ; cache)
#     bt = ITensorNetworks.bond_tensor(ψ_vidal, e)
#     ee = 0
#     for d in diag(bt)
#         ee -= abs(d) >= eps(eltype(bt)) ? d * d * log2(d * d) : 0
#     end
#     return abs(ee)
# end