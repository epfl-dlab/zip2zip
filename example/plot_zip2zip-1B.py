from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import os

SOURCE_PRETTY_NAMES = {
    "AI-MO/NuminaMath-1.5": "Math",
    "HuggingFaceFW/fineweb-2": "Multilingual",
    "HuggingFaceFW/fineweb-edu": "Knowledge",
    "devngho/the-stack-llm-annotations-v2": "Code",
    "HuggingFaceH4/ultrachat_200k": "Chat",
}


def load_and_preprocess_dataset():
    """Load the dataset and preprocess it by filtering and calculating frequencies."""
    z2z_dataset = load_dataset("epfl-dlab/zip2zip-1B")
    print(z2z_dataset)

    dataset_df = pd.DataFrame(
        {
            "source": z2z_dataset["train"]["source"],
            "token_count": z2z_dataset["train"]["token_count"],
        }
    )

    # Filter out the samples with token_count less than 1024
    dataset_df = dataset_df[dataset_df["token_count"] > 1023]
    print(dataset_df)

    return dataset_df


def plot_data(dataset_df, output_dir="plots/data_plots"):
    """Generate plots for the dataset and save them to the specified directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the total token count for each source using groupby and sum
    token_count_by_source = dataset_df.groupby("source")["token_count"].sum()

    # Map raw source names to pretty names
    pretty_labels = [
        SOURCE_PRETTY_NAMES.get(source, source)
        for source in token_count_by_source.index
    ]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1, 2]}
    )

    # Plot pie chart for the proportion of each data source by token count
    ax1.pie(
        token_count_by_source.values,
        labels=pretty_labels,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax1.set_title("Data Mixture of Zip2Zip-1B")
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Plot cumulative distribution of token_count for each source
    for source in dataset_df["source"].unique():
        source_data = dataset_df[dataset_df["source"] == source]
        sorted_token_counts = source_data["token_count"].sort_values()
        cumulative = sorted_token_counts.cumsum()
        cumulative_percentage = (
            100 * cumulative / cumulative.iloc[-1]
        )  # Convert to percentage
        pretty_name = SOURCE_PRETTY_NAMES.get(source, source)
        ax2.plot(sorted_token_counts, cumulative_percentage, label=pretty_name)

    ax2.set_title("Cumulative Distribution of Sequence Length")
    ax2.set_xlabel("Sequence Length")
    ax2.set_xscale("log")  # Set x-axis to log scale
    ax2.set_ylabel("Cumulative Percentage (%)")
    ax2.legend()
    # ax2.grid(True)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_mixture_of_zip2zip-1B.png"))
    plt.show()


def main():
    dataset_df = load_and_preprocess_dataset()
    plot_data(dataset_df)


if __name__ == "__main__":
    main()
