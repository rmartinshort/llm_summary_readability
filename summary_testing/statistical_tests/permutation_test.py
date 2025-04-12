import numpy as np
from typing import Callable, Optional

def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    difference_statistic: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    seed: int = 42,
    alternative: str = "greater",
    additional_uncertainty: Optional = None,
) -> dict:
    """
    Performs a paired permutation test to evaluate the difference between two related samples.

    Args:
        x (np.ndarray): The first sample data.
        y (np.ndarray): The second sample data.
        difference_statistic (Callable[[np.ndarray], float]): A function that computes the statistic of interest from the paired differences.
        n_permutations (int, optional): The number of permutations to perform. Defaults to 1000.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        alternative (str, optional): The alternative hypothesis to test. Can be "greater", "less", or "two-sided". Defaults to "greater".
        additional_uncertainty (Optional[dict], optional): A dictionary containing additional uncertainties for x and y, with keys "x" and "y". Defaults to None.

    Returns:
        dict: A dictionary containing the observed test statistic, p-value, and the null distribution of test statistics.
    """
    paired_differences = y - x
    observed_statistic = difference_statistic(paired_differences)
    null_distribution = []
    random_state = np.random.RandomState(seed)

    # if the two distributions have different uncertainties
    if isinstance(additional_uncertainty, dict):
        x_std = additional_uncertainty["x"]
        y_std = additional_uncertainty["y"]
        # standard deviation of combined distribution
        total_std = np.sqrt(x_std**2 + y_std**2)
    # if the two distributions have the same uncertainty
    elif additional_uncertainty:
        x_std, y_std = additional_uncertainty, additional_uncertainty
        total_std = np.sqrt(x_std**2 + y_std**2)
    # if we are not trying to take additional uncertainty into account
    else:
        total_std = 0

    for _ in range(n_permutations):
        permuted_diffs = np.array(
            [diff * random_state.choice([-1, 1]) for diff in paired_differences]
        ).astype(float)
        if total_std > 0:
            noise_term = np.random.normal(
                loc=0, scale=total_std, size=len(paired_differences)
            )
            permuted_diffs += noise_term
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

    return {
        "test_statistic": observed_statistic,
        "p_value": p_value,
        "null_distribution": null_distribution,
    }



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
