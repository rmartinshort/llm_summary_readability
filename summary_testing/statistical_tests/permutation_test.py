import numpy as np


def paired_permutation_test(
    x, y, difference_statistic, n_permutations=1000, seed=42, alternative="greater"
):
    paired_differences = y - x
    observed_statistic = difference_statistic(paired_differences)
    null_distribution = []
    random_state = np.random.RandomState(seed)

    for _ in range(n_permutations):
        permuted_diffs = np.array(
            [diff * random_state.choice([-1, 1]) for diff in paired_differences]
        ).astype(float)
        perm_stat = difference_statistic(permuted_diffs)
        null_distribution.append(perm_stat)

    if alternative == "greater":
        p_value = (
            sum(stat >= observed_statistic for stat in null_distribution)
            / n_permutations
        )
    elif alternative == "less":
        p_value = (
            sum(stat <= observed_statistic for stat in null_distribution)
            / n_permutations
        )
    else:
        p_value = (
            sum(abs(stat) >= abs(observed_statistic) for stat in null_distribution)
            / n_permutations
        )
    return observed_statistic, p_value, null_distribution


def interpret_permutation_test(statistic, p_value, p_value_threshold):
    print(
        f"The mean readability change is {statistic:.2f} (positive means increasing readability)"
    )
    print(f"The paired permutation test p value is {p_value:.3f}")

    if p_value > p_value_threshold:
        print(
            "This means we fail to reject the null hypothesis that there is no difference in readability"
        )
    else:
        if statistic > 0:
            print(
                "This means we reject the null hypothesis and conclude that there is a positive change in readability"
            )
        else:
            print(
                "This means we reject the null hypothesis and conclude that there is a negative change in readability"
            )
