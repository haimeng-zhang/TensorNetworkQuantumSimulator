using ITensors: inds, uniqueinds, eachindval, norm
using Dictionaries: set!, AbstractDictionary
using TensorNetworkQuantumSimulator: message_diff

function update_message(alpha, beta, message, belief_alpha, belief_beta, b_beta; rate = 1.0)
    marginalized_alpha = copy(belief_alpha)
    inds = uniqueinds(belief_alpha, belief_beta)
    for ind in inds
        marginalized_alpha = marginalized_alpha * ITensor(1.0, ind)
    end

    ratio = pointwise_division_raise(marginalized_alpha, belief_beta; power = rate /b_beta)
    return special_multiply(ratio, message)
end

function update_alpha_belief(alpha, psi_alpha, msgs, cs, ps)
    belief = copy(psi_alpha)
    for beta in cs[alpha]
        for parent_alpha in ps[beta]
            if parent_alpha != alpha
                belief = special_multiply(belief, msgs[(parent_alpha, beta)])
            end
        end
    end
    return belief / real(sum(belief))
end

function update_beta_belief(beta, psi_beta, msgs, ps, b_nos)
    belief = copy(psi_beta)
    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        belief = special_multiply(belief, n)
    end
    return belief / real(sum(belief))
end

function update_messages(msgs, b_alphas, b_betas, b_nos; rate)
    new_msgs = copy(msgs)
    for (alpha, beta) in keys(msgs)
        new_msg = update_message(alpha, beta, msgs[(alpha, beta)], b_alphas[alpha], b_betas[beta], b_nos[beta]; rate)
        set!(new_msgs, (alpha, beta), new_msg)
    end
    return new_msgs
end

function update_alpha_beliefs(psi_alphas, bs, msgs, cs, ps)
    alpha_beliefs = []
    for alpha in 1:length(bs)
        alpha_belief = update_alpha_belief(alpha, psi_alphas[alpha], msgs, cs, ps)
        push!(alpha_beliefs, alpha_belief)
    end
    return alpha_beliefs
end

function update_beta_beliefs(psi_betas, ms, msgs, ps, b_nos)
    beta_beliefs = []
    for beta in 1:length(ms)
        beta_belief = update_beta_belief(beta, psi_betas[beta], msgs, ps, b_nos)
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

function TensorNetworkQuantumSimulator.normalize(msgs::AbstractDictionary)
    new_msgs = copy(msgs)
    for key in keys(msgs)
        z = real(sum(msgs[key]))
        set!(new_msgs, key, msgs[key] / z)
    end
    return new_msgs
end

function generalized_belief_propagation_V2(T::TensorNetwork, bs, ms, ps, cs, b_nos, mobius_nos; niters::Int, rate::Number)
    psi_alphas = get_psis(bs, T)
    psi_betas = get_psis(ms, T)
    msgs = initialize_messages(ms, bs, ps, T)
    b_alphas = initialize_beliefs(psi_alphas)
    b_betas = initialize_beliefs(psi_betas)

    for i in 1:niters
        new_msgs = update_messages(msgs, b_alphas, b_betas, b_nos; rate)
        new_msgs = normalize(new_msgs)

        diff = message_diffs(new_msgs, msgs)

        if i % niters == 0
            println("Finished running GBP")
            println("Average difference in messages following update - $diff")
        end

        msgs = new_msgs
        b_alphas = update_alpha_beliefs(psi_alphas, bs, msgs, cs, ps)
        b_betas = update_beta_beliefs(psi_betas, ms, msgs, ps, b_nos)
    end

    f = kikuchi_free_energy(ms, bs, b_alphas, b_betas, psi_alphas, psi_betas, mobius_nos)
    return f
end