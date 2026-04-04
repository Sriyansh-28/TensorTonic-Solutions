def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    N = len(actual_tokens)
    total_log_prob = 0.0

    for i in range(N):
        p_i = prob_distributions[i][actual_tokens[i]]
        total_log_prob += math.log(p_i)

    H = -total_log_prob / N
    return math.exp(H)