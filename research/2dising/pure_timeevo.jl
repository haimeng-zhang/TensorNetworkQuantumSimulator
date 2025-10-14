using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks: ITensorNetworks

const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

using ITensors: ITensors, ITensor, Index, onehot, mapprime, prime, inds, norm, swapinds, commonind, noprime, dag, denseblocks, delta, normalize, replaceinds, replaceind, dim, dot, sim, replacetags, uniqueind

using Combinatorics

using NamedGraphs: NamedGraphs

using KrylovKit: linsolve

using CUDA

include("ctmrg.jl")

function check_symmetry(t::ITensor, symmetry_inds::Vector{<:Index})
    err = 0
    sets = collect(unique(permutations(symmetry_inds, length(symmetry_inds))))
    for inds in sets
        err += norm(t - swapinds(t, symmetry_inds, inds))
    end
    return err / length(sets)
end

function enforce_symmetry(t_int::ITensor, symmetry_inds::Vector{<:Index})
    sets = collect(unique(permutations(symmetry_inds, length(symmetry_inds))))
    t = ITensor()
    for inds in sets
        t += swapinds(t_int, symmetry_inds, inds)
    end
    return t / length(sets)
end

function square_ising_evolution_tensor(s::Index, g::Number, dt::Number, J::Number, gz::Number = 0.0)

    n = 3
    graph = named_grid((n,n))
    sinds = siteinds("S=1/2", graph)

    v = (2,2)
    sinds[v] = Index[s]
    vns = [(3,2), (2,3), (1, 2), (2,1)]
    Rx = TensorNetworkQuantumSimulator.toitensor(("Rx", [v], -im * g*dt), sinds)
    Rz =  TensorNetworkQuantumSimulator.toitensor(("Rz", [v], -im * gz*dt), sinds)

    Rzzs = [TensorNetworkQuantumSimulator.toitensor(("Rzz", NamedGraphs.NamedEdge(v => vn), -im * 2*J*dt), sinds) for vn in vns]

    t = Rx
    t = mapprime(t * prime(Rz, tags = "Spin"), 2=>1)
    t_tags = ["u", "r", "d", "l"]
    t_inds = []
    for (i, Rzz) in enumerate(Rzzs)
        U, V = ITensors.factorize(Rzz, [s,s']; cutoff = 1e-16, tags = t_tags[i], ortho = "none")
        t =  mapprime(t * prime(U, tags = "Spin"), 2=>1)
        push!(t_inds, commonind(U, V))
    end
    t = mapprime(t * prime(Rx, tags = "Spin"), 2=>1)
    t = mapprime(t * prime(Rz, tags = "Spin"), 2=>1)

    @assert check_symmetry(t, setdiff([s,s'], inds(t))) <= 1e-6

    return t, t_inds
end

function apply_U(ψt::ITensor, U::ITensor, ψt_inds, U_inds)
    ψt = copy(ψt)
    ψt =noprime(ψt * U)
    ψtinds = Index[]
    for (ψt_ind, U_ind) in zip(ψt_inds, U_inds)
        C = ITensors.combiner(ψt_ind, U_ind, tags = ITensors.tags(ψt_ind))
        ψt = ψt * C
        push!(ψtinds, ITensors.combinedind(C))
    end

    return normalize(ψt), ψtinds
end

function main()
    tags = ["u", "r", "d", "l"]
    ψt_inds = Index(1, tags[1]), Index(1, tags[2]), Index(1, tags[3]), Index(1, tags[4])
    s = Index(2, "Spin,S=1/2")
    ψt = ITensors.state("Z+", s)* prod([onehot(i => 1) for i in ψt_inds])

    J = -1

    eltype = Float64

    ψt = adapt(Vector{eltype})(ψt)
    ψt = CUDA.cu(ψt)


    Rinit, Rmax =1,4
    niters = 20
    schedule = [(100, 0.005) for i in 1:10]

    cutoff = 1e-14
    id, oz,ox = adapt(datatype(ψt))(ITensors.op("I", s)), adapt(datatype(ψt))(ITensors.op("Z", s)), adapt(datatype(ψt))(ITensors.op("X", s))
    N_ctmrg = CTMRG_ITN(ITensor[ψt, id, prime(dag(ψt))], [[i, j] for (i,j) in zip(ψt_inds, dag.(prime.(ψt_inds)))], Rinit)
    N_ctmrg = update(N_ctmrg, niters; maxdim = 1, tol = 1e-7, cutoff, alg = "svd")

    sz, sx =  effective_scalar(N_ctmrg; centre_tensors = ITensor[ψt, oz, prime(dag(ψt))]) / effective_scalar(N_ctmrg), effective_scalar(N_ctmrg; centre_tensors = ITensor[ψt, ox, prime(dag(ψt))]) / effective_scalar(N_ctmrg)
    @show sz, sx
    χ = 2

    gz = 0.001

    gs = [-3.1]

    for g in gs
        println("G is $(g)")
        effective_time = 0
        for (era, (ntime_steps, dt)) in enumerate(schedule)
            println("In Era $(era)")
            println("Effective time is $(effective_time)")
            U, U_inds = square_ising_evolution_tensor(s, g, dt, J, gz) 
            U = adapt(Vector{eltype})(U)
            U = CUDA.cu(U)
            N_ctmrg = CTMRG_ITN(ITensor[ψt, id, prime(dag(ψt))], [[i, j] for (i,j) in zip(ψt_inds, dag.(prime.(ψt_inds)))], N_ctmrg)
            sz, sx =  effective_scalar(N_ctmrg; centre_tensors = ITensor[ψt, oz, prime(dag(ψt))]) / effective_scalar(N_ctmrg), effective_scalar(N_ctmrg; centre_tensors = ITensor[ψt, ox, prime(dag(ψt))]) / effective_scalar(N_ctmrg)
            @show sz, sx
            for i in 1:ntime_steps
                ψtpdt, ψtptinds = apply_U(ψt, U, ψt_inds, U_inds)
                ψtpdt = enforce_symmetry(ψtpdt, ψtptinds)
                @assert check_symmetry(ψtpdt, ψtptinds) <= 1e-6

                N_ctmrg = CTMRG_ITN(ITensor[ψtpdt, id, prime(dag(ψtpdt))], [[i, j] for (i,j) in zip(ψtptinds, dag.(prime.(ψtptinds)))], N_ctmrg)
                #N_ctmrg = update(N_ctmrg, niters; maxdim = 2, tol = 1e-12, cutoff, alg = "svd")
                if environment_dim(N_ctmrg) < Rmax
                    N_ctmrg = update(N_ctmrg, 3; maxdim = Rmax, cutoff, alg = "svd", verbose = false)
                end
                #N_ctmrg = update(N_ctmrg, niters; maxdim = Rmax, alg = "qr", verbose=  false)
                N_ctmrg = update(N_ctmrg, niters; maxdim = Rmax, cutoff, alg = "svd", verbose = false)

                _ind = first(ψtptinds)
                ψtpdt_mod = replaceind(copy(ψtpdt), _ind, _ind'')
                effective_rho = [get_effective_environment(N_ctmrg); [ψtpdt_mod, id, prime(dag(ψtpdt))]]
                seq = ITensorNetworks.contraction_sequence(effective_rho; alg = "optimal")
                effective_rho = ITensors.contract(effective_rho; sequence = seq)
                Q, R = ITensors.factorize(effective_rho, [_ind]; maxdim = χ, cutoff, which_decomp = "eigen", ortho = "left")
                thinned_index = commonind(Q,R)
                new_ψtptinds = []
                for (j,ind) in enumerate(ψtptinds)
                    new_thinned_index = sim(replacetags(thinned_index, ITensors.tags(thinned_index) => tags[j]))
                    Qeff = replaceind(replaceind(Q, uniqueind(Q, R), ind), thinned_index, new_thinned_index)
                    ψtpdt = ψtpdt * Qeff
                    push!(new_ψtptinds, new_thinned_index)
                end

                ψt, ψt_inds = copy(ψtpdt), new_ψtptinds
                effective_time += dt

            end
        end
    end


end

main()