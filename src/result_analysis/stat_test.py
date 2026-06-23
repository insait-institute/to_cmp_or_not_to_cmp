import numpy as np

def randomizarion_test(pairs_df, n_permutations=10000, alpha=0.05, random_state=42):
    cmp_mat = np.vstack([np.asarray(x) for x, y in pairs_df]).T
    iso_mat = np.vstack([np.asarray(y) for x, y in pairs_df]).T

    n_questions, _ = cmp_mat.shape
    rng = np.random.default_rng(random_state)

    d = cmp_mat.mean(axis=1) - iso_mat.mean(axis=1)
    observed_diff = abs(d.mean())

    swap_mat = rng.choice([-1, 1], size=(n_questions, n_permutations))
    permuted_means = (d[:, np.newaxis] * swap_mat).mean(axis=0)
    diff = np.abs(permuted_means)

    count = np.sum(diff >= observed_diff)
    p_value = (count + 1) / (n_permutations + 1)
    reject = (p_value <= alpha)

    res = {
        "H0_reject" : reject.item(),
        "p_value" : p_value.item(),
    }

    return res

def bootstrap_ci(list_values, n_bootstrap=10000, alpha=0.05, random_state=42):
    list_values = np.asarray(list_values).T.mean(axis=1)

    #print(f"Bootstrap CI: n_bootstrap={n_bootstrap}, alpha={alpha}, random_state={random_state}")
    data = np.array(list_values)
    n = len(data)
    rng = np.random.default_rng(random_state)

    resample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    
    resamples = data[resample_indices]
    
    boot_means = resamples.mean(axis=1)

    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(boot_means, lower_percentile).item()
    ci_upper = np.percentile(boot_means, upper_percentile).item()
    
    #print(f"Bootstrap CI: lower={ci_lower}, mean={data.mean()}, upper={ci_upper}")
    return (ci_lower, ci_upper)