import matplotlib.pyplot as plt
import seaborn as sns


def readability_overlap_histogram(dataset_pd, x, y):
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot histograms
    sns.histplot(
        data=dataset_pd, x=x, color="blue", label=x, kde=False, alpha=0.5, binwidth=5
    )
    sns.histplot(
        data=dataset_pd, x=y, color="orange", label=y, kde=False, alpha=0.5, binwidth=5
    )

    # Calculate statistics
    x_mean = dataset_pd[x].mean()
    x_std = dataset_pd[x].std()
    y_mean = dataset_pd[y].mean()
    y_std = dataset_pd[y].std()

    # Add mean and standard deviation lines for input
    plt.axvline(x=x_mean, color="blue", linestyle="--", label=f"{x} Mean: {x_mean:.2f}")
    plt.axvline(
        x=x_mean + x_std,
        color="blue",
        linestyle=":",
        label=f"{x} +1 Std: {x_mean + x_std:.2f}",
    )
    plt.axvline(
        x=x_mean - x_std,
        color="blue",
        linestyle=":",
        label=f"{x} -1 Std: {x_mean - x_std:.2f}",
    )

    # Add mean and standard deviation lines for summary
    plt.axvline(
        x=y_mean, color="orange", linestyle="--", label=f"{y} Mean: {y_mean:.2f}"
    )
    plt.axvline(
        x=y_mean + y_std,
        color="orange",
        linestyle=":",
        label=f"{y} +1 Std: {y_mean + y_std:.2f}",
    )
    plt.axvline(
        x=y_mean - y_std,
        color="orange",
        linestyle=":",
        label=f"{y} -1 Std: {y_mean - y_std:.2f}",
    )

    # Add labels and legend
    plt.title("Overlapping Histograms of Flesch Reading Ease")
    plt.xlabel("Flesch Reading Ease")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()


def readability_change_histogram(dataset_pd, x, y):
    dataset_pd["readability_change"] = dataset_pd[y] - dataset_pd[x]
    sns.histplot(x="readability_change", data=dataset_pd)
    input_mean = dataset_pd["readability_change"].mean()
    input_std = dataset_pd["readability_change"].std()
    plt.axvline(
        x=input_mean, color="blue", linestyle="--", label=f"Mean: {input_mean:.2f}"
    )
    plt.axvline(
        x=input_mean + input_std,
        color="blue",
        linestyle=":",
        label=f"Mean +1 Std: {input_mean + input_std:.2f}",
    )
    plt.axvline(
        x=input_mean - input_std,
        color="blue",
        linestyle=":",
        label=f"Mean -1 Std: {input_mean - input_std:.2f}",
    )
    plt.xlabel(f"Readability change ({y} - {x})")
    plt.title("Change in Flesch Reading Ease")
    plt.legend(bbox_to_anchor=(0.9, 1))
    plt.tight_layout()
