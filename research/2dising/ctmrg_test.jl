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

using ITensorNetworks: AbstractBeliefPropagationCache

using NamedGraphs.PartitionedGraphs: PartitionEdge

using Combinatorics

using Dictionaries

using TensorOperations: TensorOperations, @tensor

# pkg> add QuadGK
using QuadGK

function phi_isingsq(K::Real)
    s2 = sinh(2K)
    c2 = cosh(2K)
    k  = 2*s2 / (c2*c2)
    k2 = min(k*k, 1 - 1e-15)   # guard against tiny overshoot near Kc

    integrand(θ) = log(0.5*(1 + sqrt(1 - k2*sin(θ)^2)))
    I, _ = quadgk(integrand, 0.0, π/2; rtol=1e-12, atol=1e-14)

    return log(2*c2) + (1/π)*I
end


f_isingsq(K::Real; J::Real=1.0) = -(J/K) * phi_isingsq(K)


function ising_tensor(K)
    # bond weight matrix
    t = [exp(K) exp(-K);
        exp(-K) exp(K)]

    # diagonalize
    r = eigen(t)
    nt = r.vectors * LinearAlgebra.Diagonal(sqrt.(r.values)) * r.vectors'  # nt is 2x2

    # define indices
    s1 = Index(2, "s1")
    s2 = Index(2, "s2")
    s3 = Index(2, "s3")
    s4 = Index(2, "s4")

    # local partition function tensor (delta on all spins equal)
    O = ITensor(s1, s2, s3, s4)
    O[s1=>1, s2=>1, s3=>1, s4=>1] = 1.0
    O[s1=>2, s2=>2, s3=>2, s4=>2] = 1.0

    # bond matrices nt, as ITensors
    i1 = Index(2,"i1")
    i2 = Index(2,"i2")
    i3 = Index(2,"i3")
    i4 = Index(2,"i4")

    NT1 = ITensor(nt, i1, s1)
    NT2 = ITensor(nt, i2, s2)
    NT3 = ITensor(nt, i3, s3)
    NT4 = ITensor(nt, i4, s4)

    # contract everything
    o = O * NT1 * NT2 * NT3 * NT4
    # o is an ITensor with indices (i1,i2,i3,i4)
    return o, i1, i2, i3, i4
end

using Random
function main()

    Random.seed!(2984)
    s = Index(2, "S=1/2")
    β =0.25
    t1, l, r, u, d = ising_tensor(β)
    
    Rinit, Rmax =2,10
    ψ0 = CTMRG_ITN(ITensor[t1], [[u], [r], [d], [l]], Rinit)

    niters = 20
    ψ1 = update(ψ0, niters; maxdim = Rinit, cutoff = 1e-12, alg = "svd")
    ψ1 = update(ψ1, niters; maxdim = Rmax, cutoff = 1e-12, alg = "svd")

    @show environment_dim(ψ1)
    z =  scalar(ψ1)

    @show abs(log(z) - phi_isingsq(β))
end

main()