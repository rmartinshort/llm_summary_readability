import numpy as np
from typing import Callable, Optional

def paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    difference_statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    additional_uncertainty: Optional = None
) -> dict:
    """
    Computes the paired bootstrap confidence interval for the difference between two related samples.

    Args:
        x (np.ndarray): The first sample data.
        y (np.ndarray): The second sample data.
        difference_statistic (Callable[[np.ndarray], float]): A function that computes the statistic of interest from the paired differences.
        n_bootstrap (int, optional): The number of bootstrap resamples to perform. Defaults to 10000.
        ci_level (float, optional): The confidence level for the interval. Defaults to 0.95.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        additional_uncertainty (Optional[dict], optional): A dictionary containing additional uncertainty information with keys "x" and "y". Defaults to None.

    Returns:
        dict: A dictionary containing the observed test statistic, the confidence interval, and the bootstrap statistics.
    """
    paired_differences = y - x
    observed_statistic = difference_statistic(paired_differences)
    n_pairs = len(paired_differences)
    bootstrap_stats = []
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

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = random_state.choice(n_pairs, size=n_pairs, replace=True)
        bootstrap_sample = paired_differences[indices]

        # add extra uncertainty associated with LLM calls if desired
        if total_std > 0:
            noise_term = np.random.normal(
                loc=0, scale=total_std, size=len(bootstrap_sample)
            )
            bootstrap_sample += noise_term

        # Calculate and store the bootstrap statistic
        bootstrap_mean = difference_statistic(bootstrap_sample)
        bootstrap_stats.append(bootstrap_mean)

    # Calculate confidence interval
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return {
        "test_statistic": observed_statistic,
        "confidence_interval": (ci_lower, ci_upper),
        "bootstrap_stats": bootstrap_stats,
    }



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
