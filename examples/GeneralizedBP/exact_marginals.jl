function kikuchi_free_energy(ms, bs, b_alphas, b_betas, psi_alphas, psi_betas, mobius_nos)
    f = 0
    for alpha in 1:length(bs)
        R = pointwise_division_raise(b_alphas[alpha], psi_alphas[alpha])
        R = elementwise_operation(x -> x > 1e-12 ? log(x) : 0, R)
        R = special_multiply(R, b_alphas[alpha])
        f += sum(R)
    end

    for beta in 1:length(ms)
        R = pointwise_division_raise(b_betas[beta], psi_betas[beta])
        R = elementwise_operation(x -> x > 1e-12 ? log(x) : 0, R)
        R = special_multiply(R, b_betas[beta])
        f += mobius_nos[beta] * sum(R)
    end

    return f
end