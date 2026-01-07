import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from manta.utils.analysis import get_dominant_topics


def gen_violin_plot(W, S_matrix, datetime_series, table_output_dir, table_name):
    # Get dominant topics for each document
    W_matrix = W
    S_matrix = S_matrix if S_matrix is not None else None
    n_topics = W_matrix.shape[1]
    dominant_topics = get_dominant_topics(W_matrix, min_score=0.0, s_matrix=S_matrix)

    # Extract year from datetime series
    years = datetime_series.dt.year

    # Prepare data: create weighted year distribution for each topic based on document counts
    violin_data = []
    for doc_idx in range(len(W_matrix)):
        year = int(years.iloc[doc_idx])
        dominant_topic_idx = dominant_topics[doc_idx]

        if dominant_topic_idx != -1:  # Only include valid topics
            topic_id = dominant_topic_idx + 1
            violin_data.append({"Topic": topic_id, "Year": year})

    violin_df = pd.DataFrame(violin_data)

    # Convert Topic column to ordered categorical for numerical sorting
    # This ensures "Topic 2" comes before "Topic 10" instead of lexicographic ordering
    unique_topics = sorted(violin_df["Topic"].unique())
    topic_labels = [f"Topic {t}" for t in unique_topics]
    violin_df["Topic"] = pd.Categorical(
        [f"Topic {t}" for t in violin_df["Topic"]],
        categories=topic_labels,
        ordered=True
    )

    # Get unique topics for plot sizing
    n_topics_found = violin_df["Topic"].nunique()

    # Create horizontal violin plot: one violin per topic showing year distribution
    fig_violin, ax = plt.subplots(figsize=(12, max(8, n_topics_found * 0.8)))

    sns.violinplot(
        data=violin_df, y="Topic", x="Year", orient="h", inner="box", palette="Set2", ax=ax
    )

    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Topic ID", fontsize=12, fontweight="bold")
    ax.set_title("Topic Distribution Across Years", fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save the plot
    violin_path = table_output_dir / f"{table_name}_topic_distribution_by_year.png"
    fig_violin.savefig(violin_path, dpi=300, bbox_inches="tight")
    plt.close(fig_violin)

    return violin_path
