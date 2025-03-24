import numpy as np


def paired_bootstrap_ci(
    x, y, difference_statistic, n_bootstrap=10000, ci_level=0.95, seed=42
):
    paired_differences = y - x
    observed_statistic = difference_statistic(paired_differences)
    n_pairs = len(paired_differences)
    bootstrap_stats = []
    random_state = np.random.RandomState(seed)

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = random_state.choice(n_pairs, size=n_pairs, replace=True)
        bootstrap_sample = paired_differences[indices]

        # Calculate and store the bootstrap statistic
        bootstrap_mean = difference_statistic(bootstrap_sample)
        bootstrap_stats.append(bootstrap_mean)

    # Calculate confidence interval
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return observed_statistic, (ci_lower, ci_upper), bootstrap_stats


def interpret_bootstrap_ci_intervals(statistic, ci_lower, ci_upper):
    print(
        f"The mean readability change is {statistic:.2f} (positive means increasing readability)"
    )
    print(f"The paired bootstrap confidence range is {ci_lower:.3f} to {ci_upper:.3f}")

    if ci_lower > 0:
        print(
            "This suggests we should reject the null hypothesis and conclude there was a meaningful positive change in readability"
        )
    elif ci_upper < 0:
        print(
            "This suggests we should reject the null hypothesis and conclude there was a meaningful negative change in readability"
        )
    else:
        print(
            "This suggests we should not reject the null hypothesis that there is no difference in readability"
        )
