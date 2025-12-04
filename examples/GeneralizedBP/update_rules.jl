using ITensors: inds, uniqueinds, eachindval, norm
using Dictionaries: set!
using TensorNetworkQuantumSimulator: message_diff

function pointwise_division_raise(a::ITensor, b::ITensor; power = 1)
    @assert Set(inds(a)) == Set(inds(b))
    out = ITensor(1.0, inds(a))
    indexes = inds(a)
    for iv in eachindval(out)
        out[iv...] = (a[iv...] / b[iv...])^(power)
    end

    return out
end

function raise(tensor::ITensor, power::Number)
    out = ITensor(1.0, inds(tensor))
    for iv in eachindval(out)
        out[iv...] = tensor[iv...]^power
    end
    return out
end

function update_n_message(alpha, beta, message, belief_alpha, belief_beta, b_beta; rate = 1.0)
    marginalized_alpha = copy(belief_alpha)
    inds = uniqueinds(belief_alpha, belief_beta)
    for ind in inds
        marginalized_alpha = marginalized_alpha * ITensor(1.0, ind)
    end

    ratio = pointwise_division_raise(marginalized_alpha, belief_beta; power = rate /b_beta)
    return special_multiply(ratio, message)
end

function update_m_message(alpha, beta, ps, n_msgs)
    parent_alphas = ps[beta]
    m_msg = ITensor(1.0, inds(n_msgs[(alpha, beta)]))
    for parent_alpha in parent_alphas
        if parent_alpha != alpha
            m_msg = special_multiply(m_msg, n_msgs[(parent_alpha, beta)])
        end
    end
    return m_msg
end

function update_alpha_belief(alpha, psi_alpha, m_msgs, cs)
    belief = copy(psi_alpha)
    for beta in cs[alpha]
        belief = special_multiply(belief, m_msgs[(alpha, beta)])
    end
    return belief / real(sum(belief))
end

function update_beta_belief(beta, psi_beta, n_msgs, ps, b_nos)
    belief = copy(psi_beta)
    for alpha in ps[beta]
        n = raise(n_msgs[(alpha, beta)], b_nos[beta])
        belief = special_multiply(belief, n)
    end
    return belief / real(sum(belief))
end

function update_n_messages(n_msgs, b_alphas, b_betas, b_nos; rate)
    new_n_msgs = copy(n_msgs)
    for (alpha, beta) in keys(n_msgs)
        new_n_msg = update_n_message(alpha, beta, n_msgs[(alpha, beta)], b_alphas[alpha], b_betas[beta], b_nos[beta]; rate)
        set!(new_n_msgs, (alpha, beta), new_n_msg)
    end
    return new_n_msgs
end

function update_m_messages(n_msgs, ps)
    m_msgs = copy(n_msgs)
    for (alpha, beta) in keys(n_msgs)
        set!(m_msgs, (alpha, beta), update_m_message(alpha, beta, ps, n_msgs))
    end
    return m_msgs
end

function update_alpha_beliefs(psi_alphas, bs, m_msgs, cs)
    alpha_beliefs = []
    for alpha in 1:length(bs)
        alpha_belief = update_alpha_belief(alpha, psi_alphas[alpha], m_msgs, cs)
        push!(alpha_beliefs, alpha_belief)
    end
    return alpha_beliefs
end

function update_beta_beliefs(psi_betas, ms, n_msgs, ps, b_nos)
    beta_beliefs = []
    for beta in 1:length(ms)
        beta_belief = update_beta_belief(beta, psi_betas[beta], n_msgs, ps, b_nos)
        push!(beta_beliefs, beta_belief)
    end
    return beta_beliefs
end

function message_diffs(msgs1, msgs2)
    @assert keys(msgs1) == keys(msgs2)
    diff = 0
    for key in keys(msgs1)
        diff += message_diff(msgs1[key], msgs2[key])
    end
    return diff / length(keys(msgs1))
end

function normalize(msgs)
    new_msgs = copy(msgs)
    for key in keys(msgs)
        z = real(sum(msgs[key]))
        set!(new_msgs, key, msgs[key] / z)
    end
    return new_msgs
end

function generalized_belief_propagation(T::TensorNetwork, bs, ms, ps, cs, b_nos; niters::Int, rate::Number)
    println("Running Generalized Belief Propagation on the norm of a 10 x 10 random Tensor Network State")
    psi_alphas = get_psis(bs, T)
    psi_betas = get_psis(ms, T)
    m_msgs = initialize_messages(ms, bs, ps)
    n_msgs = copy(m_msgs)
    b_alphas = initialize_beliefs(psi_alphas)
    b_betas = initialize_beliefs(psi_betas)

    for i in 1:niters
        new_n_msgs = update_n_messages(n_msgs, b_alphas, b_betas, b_nos; rate)
        new_m_msgs = update_m_messages(n_msgs, ps)
        
        new_n_msgs = normalize(new_n_msgs)
        new_m_msgs = normalize(new_m_msgs)

        n_diff = message_diffs(new_n_msgs, n_msgs)
        m_diff = message_diffs(new_m_msgs, m_msgs)

        if i % 25 == 0 || i == 2
            println("On GBP iteration $i")
            println("Average difference in messages following update - n_msgs: $n_diff, m_msgs: $m_diff")
        end

        n_msgs = new_n_msgs
        m_msgs = new_m_msgs
        b_alphas = update_alpha_beliefs(psi_alphas, bs, m_msgs, cs)
        b_betas = update_beta_beliefs(psi_betas, ms, n_msgs, ps, b_nos)
    end
end