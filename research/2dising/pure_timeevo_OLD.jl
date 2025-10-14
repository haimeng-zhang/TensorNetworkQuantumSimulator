using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks: ITensorNetworks

const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

using ITensors: ITensors, ITensor, Index, onehot, mapprime, prime, inds, norm, swapinds, commonind, noprime

using Combinatorics

using NamedGraphs: NamedGraphs

using KrylovKit: linsolve

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

function square_ising_evolution_tensor(s::Index, g::Number, dt::Number, J::Number = 1.0)

    n = 3
    graph = named_grid((n,n))
    sinds = siteinds("S=1/2", graph)

    v = (2,2)
    sinds[v] = Index[s]
    vns = [(3,2), (2,3), (1, 2), (2,1)]
    Rx = TensorNetworkQuantumSimulator.toitensor(("Rx", [v], -im * g*dt), sinds)

    Rzzs = [TensorNetworkQuantumSimulator.toitensor(("Rzz", NamedGraphs.NamedEdge(v => vn), -im * 2*J*dt), sinds) for vn in vns]

    t = Rx
    t_tags = ["u", "r", "d", "l"]
    t_inds = []
    for (i, Rzz) in enumerate(Rzzs)
        U, V = ITensors.factorize(Rzz, [s,s']; cutoff = 1e-16, tags = t_tags[i], ortho = "none")
        t =  mapprime(t * prime(U, tags = "Spin"), 2=>1)
        push!(t_inds, commonind(U, V))
    end
    t = mapprime(t * prime(Rx, tags = "Spin"), 2=>1)

    @assert check_symmetry(t, setdiff([s,s'], inds(t))) <= 1e-14

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

    g, dt, J = -2.0, -0.2, -1
    U, U_inds = square_ising_evolution_tensor(s, g, dt, J) 

    Rinit, Rmax =1,3
    niters = 20
    ntime_steps = 2

    cutoff = 1e-12
    N_ctmrg = CTMRG_ITN(ITensor[ψt, ITensors.op("I", s), prime(dag(ψt))], [[i, j] for (i,j) in zip(ψt_inds, dag.(prime.(ψt_inds)))], Rinit)
    N_ctmrg = update(N_ctmrg, niters; maxdim = 1, tol = 1e-12, cutoff, alg = "svd")

    sz = effective_scalar(N_ctmrg) / effective_scalar(N_ctmrg; centre_tensors = ITensor[ψt, ITensors.op("Z", s), prime(dag(ψt))])
    @show sz

    n_internal_iters = 40
    χ = 2

    M_ctmrg = nothing
    for i in 1:ntime_steps
        ψtpdt, ψtptinds = apply_U(ψt, U, ψt_inds, U_inds)
        @assert check_symmetry(ψtpdt, ψtptinds) <= 1e-14

        #ψtpdt_guess, ψtpdt_guess_inds = copy(ψt), ψt_inds

        ψtpdt_guess, ψtpdt_guess_inds = enforce_symmetry(random_itensor(inds(ψt)), [i for i in ψt_inds]), ψt_inds

        if first(dim.(ψtptinds)) <= χ
            ψt, ψt_inds = copy(ψtpdt), ψtptinds
        else
            for j in 1:n_internal_iters
                M_ctmrg = CTMRG_ITN(ITensor[prime(dag(ψtpdt)), ITensors.op("I", s), ψtpdt_guess], [[i, j] for (i,j) in zip(prime.(dag.(ψtptinds)), ψtpdt_guess_inds)], Rinit)
                M_ctmrg = update(M_ctmrg, niters; maxdim = Rmax, tol = 1e-12, cutoff = 1e-12, alg = "svd")

                N_ctmrg = CTMRG_ITN(ITensor[ψtpdt_guess, ITensors.op("I", s), prime(dag(ψtpdt_guess))], [[i, j] for (i,j) in zip(ψtpdt_guess_inds, dag.(prime.(ψtpdt_guess_inds)))], Rinit)
                N_ctmrg = update(N_ctmrg, niters; maxdim = Rmax, tol = 1e-12, cutoff, alg = "svd")

                b_ts = [get_effective_environment(M_ctmrg); ITensor[dag(prime(ψtpdt)),  ITensors.op("I", s)]]
                b = ITensors.contract(b_ts; sequence = ITensorNetworks.contraction_sequence(b_ts; alg = "optimal"))
                N_ts = [get_effective_environment(N_ctmrg); ITensor[ITensors.op("I", s)]]
                seq = ITensorNetworks.contraction_sequence([ITensor[dag(prime(ψtpdt_guess))]; N_ts]; alg = "optimal")
                f = x -> ITensors.contract([ITensor[dag(prime(x))]; N_ts]; sequence = seq)
                ψtpdt_guess_new, info = linsolve(f, b, ψtpdt_guess)

                ψtpdt_guess_new = enforce_symmetry(ψtpdt_guess_new, ψtpdt_guess_inds)

                ψtpdt_guess_new = normalize(ψtpdt_guess_new)
                ψtpdt_guess = normalize(ψtpdt_guess)
                @show 1 - abs2(dot(ψtpdt_guess_new, ψtpdt_guess))
                ψtpdt_guess = copy(ψtpdt_guess_new)
                #@show inds(b)
                #@show inds(ψtpdt_guess)
                #@show inds(f(ψtpdt_guess))
            end
            ψt, ψt_inds = copy(ψtpdt_guess), ψtpdt_guess_inds
        end

        # sz =  effective_scalar(N_ctmrg; centre_tensors = ITensor[ψtpdt, ITensors.op("Z", s), prime(dag(ψtpdt))]) / effective_scalar(N_ctmrg)

        # @show sz
        # #@show environment_dim(M_ctmrg)
        # @show environment_dim(N_ctmrg)
    end


end

main()