#!/usr/bin/env python3
"""
Custom Topic Distribution Bar Plot with Custom Topic Names.

This script regenerates the topic distribution bar chart with custom topic names
instead of the default "Topic 1", "Topic 2", etc.

Usage:
    1. Edit the TOPIC_NAMES dictionary at the bottom of this script
    2. Set the NPZ_PATH to your model_components.npz file
    3. Run: uv run python tools/custom_topic_dist_plot.py
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def get_dominant_topics(W: np.ndarray, min_score: float = 0.0) -> np.ndarray:
    """Get the dominant topic for each document."""
    dominant_topics = np.argmax(W, axis=1)
    max_scores = np.max(W, axis=1)
    dominant_topics[max_scores < min_score] = -1
    return dominant_topics


def create_topic_dist_plot(
    npz_path: str,
    topic_names: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 7),
    dpi: int = 300,
    title: str = "Number of Documents per Topic",
    bar_color: str = "#1f77b4",
    show_counts: bool = True,
) -> None:
    """
    Create a topic distribution bar chart with custom topic names.

    Args:
        npz_path: Path to the model_components.npz file.
        topic_names: Dictionary mapping topic numbers to custom names.
                     Example: {1: "Healthcare Policy", 2: "Medical Education", ...}
        output_path: Path to save the output PNG. If None, saves next to npz file.
        figsize: Figure size as (width, height) tuple.
        dpi: Resolution of output image.
        title: Plot title.
        bar_color: Color for the bars.
        show_counts: Whether to show count labels on top of bars.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Load model components
    print(f"Loading model components from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    W = data["W"]
    print(f"Loaded W matrix: {W.shape[0]} documents × {W.shape[1]} topics")

    n_topics = W.shape[1]

    # Get dominant topics
    dominant_topics = get_dominant_topics(W, min_score=0.0)

    # Filter out documents with no dominant topic
    valid_mask = dominant_topics != -1
    valid_dominant_topics = dominant_topics[valid_mask]
    excluded_count = np.sum(~valid_mask)

    if excluded_count > 0:
        print(f"Excluded {excluded_count} documents with zero topic scores")

    # Count documents per topic
    topic_counts = np.bincount(valid_dominant_topics, minlength=n_topics)

    # Create labels from topic_names dictionary
    labels = []
    for i in range(n_topics):
        topic_num = i + 1  # 1-indexed
        label = topic_names.get(topic_num, f"Topic {topic_num}")
        labels.append(label)

    # Print counts
    print("\nNumber of documents per topic:")
    for i, (label, count) in enumerate(zip(labels, topic_counts)):
        print(f"  {label}: {count} documents")

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    x_positions = np.arange(n_topics)
    bars = ax.bar(x_positions, topic_counts, color=bar_color, edgecolor="black", linewidth=0.5)

    # Add count labels on top of bars
    if show_counts:
        for i, (bar, count) in enumerate(zip(bars, topic_counts)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(topic_counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Styling
    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("Number of Documents", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set x-axis labels with custom names
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Add grid on y-axis only
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem.replace('_model_components', '')}_topic_dist_custom_names.png"
    else:
        output_path = Path(output_path)

    # Save plot
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved to: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    # =========================================================================
    # EDITABLE SECTION - Customize your topic names here
    # =========================================================================

    # Map topic numbers to custom names
    # Edit the values (right side) to your desired topic names
    TOPIC_NAMES = {
        1: "Curriculum Development",
        2: "Assessment Methods",
        3: "Residency Programs",
        4: "Systematic Reviews",
        5: "ACGME Competencies",
        6: "Clinical Practice",
        7: "AI and LLMs",
        8: "Psychometric Validation",
        9: "Equity and Inclusion",
        10: "Digital Learning",
    }

    # Path to the model_components.npz file
    NPZ_PATH = "/Users/emirkarayagiz/Work/nmf-standalone/results/bildiri/TopicAnalysis/Output/medicaleducation_pnmf_bpe_10/medicaleducation_pnmf_bpe_10_model_components.npz"

    # Optional: Custom output path (set to None to auto-generate)
    OUTPUT_PATH = None

    # Optional: Custom plot title
    TITLE = "Number of Documents per Topic"

    # =========================================================================
    # END EDITABLE SECTION
    # =========================================================================

    create_topic_dist_plot(
        npz_path=NPZ_PATH,
        topic_names=TOPIC_NAMES,
        output_path=OUTPUT_PATH,
        title=TITLE,
    )
