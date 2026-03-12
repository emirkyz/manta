#!/usr/bin/env python3
"""
Custom t-SNE Topic Visualization with Custom Topic Names.

This script regenerates the t-SNE visualization with custom topic names
instead of the default "Topic 1", "Topic 2", etc.

Usage:
    1. Edit the TOPIC_NAMES dictionary at the bottom of this script
    2. Set the NPZ_PATH to your model_components.npz file
    3. Run: uv run python tools/custom_tsne_plot.py
"""

import colorsys
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def _generate_distinct_colors(n_topics: int) -> list:
    """Generate maximally distinct colors for topics."""
    import matplotlib.colors as mcolors

    distinct_palettes = {
        2: ["#E31A1C", "#1F78B4"],
        3: ["#E31A1C", "#33A02C", "#1F78B4"],
        4: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4"],
        5: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A"],
        6: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#FB9A99"],
        7: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#FB9A99", "#B15928"],
        8: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#FB9A99", "#B15928", "#FDBF6F"],
        9: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#FB9A99", "#B15928", "#FDBF6F", "#CAB2D6"],
        10: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#FB9A99", "#B15928", "#FDBF6F", "#CAB2D6", "#FFFF99"],
    }

    if n_topics <= 10 and n_topics in distinct_palettes:
        colors = distinct_palettes[n_topics]
        return [mcolors.hex2color(color) for color in colors]

    if n_topics <= 20:
        base_colors = plt.cm.tab20(np.arange(20))
        optimized_order = [0, 10, 2, 12, 4, 14, 6, 16, 8, 18, 1, 11, 3, 13, 5, 15, 7, 17, 9, 19]
        reordered_colors = [base_colors[i] for i in optimized_order[:n_topics]]
        return [(r, g, b, a) for r, g, b, a in reordered_colors]

    # For larger numbers, generate colors algorithmically
    colors = [(0.8, 0.2, 0.2, 1.0)]
    for i in range(1, n_topics):
        best_color = None
        best_min_distance = 0
        for _ in range(100):
            h = np.random.uniform(0, 1)
            s = np.random.uniform(0.6, 1.0)
            v = np.random.uniform(0.4, 0.9)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            candidate = (r, g, b, 1.0)
            min_distance = min(
                ((candidate[0] - c[0]) ** 2 + (candidate[1] - c[1]) ** 2 + (candidate[2] - c[2]) ** 2) ** 0.5
                for c in colors
            )
            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_color = candidate
        colors.append(best_color if best_color else (i / n_topics, 0.7, 0.7, 1.0))
    return colors


def get_dominant_topics(W: np.ndarray, min_score: float = 0.0) -> np.ndarray:
    """Get the dominant topic for each document."""
    dominant_topics = np.argmax(W, axis=1)
    max_scores = np.max(W, axis=1)
    dominant_topics[max_scores < min_score] = -1
    return dominant_topics


def _remove_outliers_percentile(tsne_data: pd.DataFrame, percentile: float = 0.95) -> pd.DataFrame:
    """Remove outlier points that are far from the main cluster."""
    if len(tsne_data) <= 10:
        return tsne_data

    center_x = tsne_data["x"].median()
    center_y = tsne_data["y"].median()
    distances = np.sqrt((tsne_data["x"] - center_x) ** 2 + (tsne_data["y"] - center_y) ** 2)
    threshold = np.quantile(distances, percentile)
    mask = distances <= threshold
    filtered_data = tsne_data[mask].reset_index(drop=True)

    n_removed = len(tsne_data) - len(filtered_data)
    if n_removed > 0:
        print(f"Removed {n_removed} outlier points (keeping {percentile*100:.1f}% closest to center)")

    return filtered_data


def create_tsne_plot(
    npz_path: str,
    topic_names: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 10),
    dpi: int = 400,
    title: str = "Topic Distribution Visualization",
    perplexity: int = 45,
    outlier_percentile: float = 0.95,
    point_size_multiplier: float = 1.0,
) -> None:
    """
    Create a t-SNE visualization with custom topic names.

    Args:
        npz_path: Path to the model_components.npz file.
        topic_names: Dictionary mapping topic numbers to custom names.
                     Example: {1: "Healthcare Policy", 2: "Medical Education", ...}
        output_path: Path to save the output PNG. If None, saves next to npz file.
        figsize: Figure size as (width, height) tuple.
        dpi: Resolution of output image.
        title: Plot title.
        perplexity: t-SNE perplexity parameter.
        outlier_percentile: Fraction of points to keep (0.95 = keep 95% closest to center).
        point_size_multiplier: Multiplier for point size.
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
    n_docs = W.shape[0]

    # Run t-SNE
    print(f"Running t-SNE (perplexity={perplexity})... This may take a while for large datasets.")
    method = "barnes_hut" if n_docs > 1000 else "exact"
    tsne = TSNE(
        random_state=42,
        perplexity=perplexity,
        n_iter=300,
        learning_rate="auto",
        method=method,
    )
    tsne_embedding = tsne.fit_transform(W)
    tsne_df = pd.DataFrame(tsne_embedding, columns=["x", "y"])

    # Get dominant topics
    dominant_topics = get_dominant_topics(W, min_score=0.0)
    tsne_df["hue"] = dominant_topics

    # Filter out documents with no dominant topic
    valid_mask = tsne_df["hue"] != -1
    excluded_count = (~valid_mask).sum()
    if excluded_count > 0:
        print(f"Excluded {excluded_count} documents with zero topic scores")
    tsne_df = tsne_df[valid_mask].reset_index(drop=True)

    # Remove outliers
    tsne_df = _remove_outliers_percentile(tsne_df, percentile=outlier_percentile)

    print(f"Plotting {len(tsne_df)} documents...")

    # Generate colors
    distinct_colors = _generate_distinct_colors(n_topics)

    # Calculate point size
    n_points = len(tsne_df)
    if n_points <= 100:
        point_size = 50
    elif n_points <= 500:
        point_size = 35
    elif n_points <= 1000:
        point_size = 25
    else:
        point_size = max(8, 25 - np.log10(max(n_points, 10)) * 4)
    point_size *= point_size_multiplier

    # Create plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white", edgecolor="none")

    # Create colormap
    from matplotlib.colors import ListedColormap

    colormap = ListedColormap(distinct_colors)

    # Scatter plot (matching Figure 3 style)
    scatter = ax.scatter(
        tsne_df["x"],
        tsne_df["y"],
        s=point_size,
        c=tsne_df["hue"],
        cmap=colormap,
        alpha=0.7,
        edgecolors="none",
        linewidths=0,
    )

    # Create legend with custom names
    unique_topics = sorted(tsne_df["hue"].unique())
    legend_handles = []
    for idx, topic_id in enumerate(unique_topics):
        color = distinct_colors[idx] if idx < len(distinct_colors) else (0.5, 0.5, 0.5, 1.0)
        # Get custom name (topic_id is 0-indexed, topic_names keys are 1-indexed)
        topic_num = topic_id + 1
        label = topic_names.get(topic_num, f"Topic {topic_num}")
        legend_handles.append(mpatches.Patch(color=color, label=label))

    # Legend below the plot
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(3, len(unique_topics)),
        fontsize=9,
        framealpha=0.9,
        title="Topics",
        title_fontsize=10,
    )

    # Styling
    ax.set_title(title, fontsize=18, fontweight="bold", pad=25, color="#2E3440", family="sans-serif")
    ax.set_xlabel("t-SNE Component 1", fontsize=13, color="#4C566A", fontweight="medium")
    ax.set_ylabel("t-SNE Component 2", fontsize=13, color="#4C566A", fontweight="medium")

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_facecolor("#ffffff")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E9F0")
    ax.spines["bottom"].set_color("#E5E9F0")

    ax.tick_params(axis="both", which="major", labelsize=10, colors="#4C566A")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem.replace('_model_components', '')}_tsne_custom_names.jpg"
    else:
        output_path = Path(output_path)

    # Save plot
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.2,
    )
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print(f"\nSummary: {len(tsne_df):,} documents, {len(unique_topics)} topics")
    for topic_id in unique_topics:
        topic_count = len(tsne_df[tsne_df["hue"] == topic_id])
        percentage = (topic_count / len(tsne_df)) * 100
        topic_num = topic_id + 1
        label = topic_names.get(topic_num, f"Topic {topic_num}")
        print(f"  {label}: {topic_count:,} documents ({percentage:.1f}%)")

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
    TITLE = "Topic Distribution Visualization"

    # =========================================================================
    # END EDITABLE SECTION
    # =========================================================================

    create_tsne_plot(
        npz_path=NPZ_PATH,
        topic_names=TOPIC_NAMES,
        output_path=OUTPUT_PATH,
        title=TITLE,
    )
