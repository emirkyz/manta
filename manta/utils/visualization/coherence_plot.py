"""
Coherence score visualization for topic count optimization.

This module provides plotting functionality to visualize how coherence scores
vary with different topic counts, helping users identify the optimal number
of topics for their dataset.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_coherence_results(
    topic_counts: List[int],
    coherence_scores: List[float],
    output_dir: str,
    nmf_method: str,
    optimal_idx: Optional[int] = None,
    elbow_idx: Optional[int] = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> Optional[str]:
    """
    Plot coherence scores vs topic counts with optimal and elbow points marked.

    Creates a line plot showing how coherence scores change with different
    numbers of topics. Marks the optimal point (highest coherence) and
    optionally the elbow point (diminishing returns).

    Args:
        topic_counts: List of evaluated topic counts.
        coherence_scores: List of corresponding coherence scores.
        output_dir: Directory to save the plot.
        nmf_method: NMF method name for plot title.
        optimal_idx: Index of optimal topic count (if None, calculated as argmax).
        elbow_idx: Index of elbow point (if None, not shown).
        save_plot: Whether to save plot to file.
        show_plot: Whether to display interactive plot.

    Returns:
        Path to saved plot file (if saved), None otherwise.
    """
    # Calculate optimal index if not provided
    if optimal_idx is None:
        optimal_idx = int(np.argmax(coherence_scores))

    optimal_topics = topic_counts[optimal_idx]
    optimal_coherence = coherence_scores[optimal_idx]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot main coherence line
    ax.plot(
        topic_counts,
        coherence_scores,
        "b-o",
        linewidth=2,
        markersize=8,
        label="Coherence Score",
    )

    # Mark optimal point (highest coherence)
    ax.plot(
        optimal_topics,
        optimal_coherence,
        "r*",
        markersize=20,
        label=f"Optimal: {optimal_topics} topics (C_V={optimal_coherence:.4f})",
        zorder=5,
    )

    # Mark elbow point if provided and different from optimal
    if elbow_idx is not None and elbow_idx != optimal_idx:
        elbow_topics = topic_counts[elbow_idx]
        elbow_coherence = coherence_scores[elbow_idx]
        ax.plot(
            elbow_topics,
            elbow_coherence,
            "g^",
            markersize=15,
            label=f"Elbow: {elbow_topics} topics (C_V={elbow_coherence:.4f})",
            zorder=5,
        )

    # Configure axes
    ax.set_xlabel("Number of Topics", fontsize=12)
    ax.set_ylabel("Gensim C_V Coherence Score", fontsize=12)
    ax.set_title(
        f"Topic Count Optimization - {nmf_method.upper()} Method",
        fontsize=14,
        fontweight="bold",
    )

    # Set x-axis ticks to show all topic counts if reasonable
    if len(topic_counts) <= 30:
        ax.set_xticks(topic_counts)
    else:
        # For large ranges, show every nth tick
        step = max(1, len(topic_counts) // 15)
        ax.set_xticks(topic_counts[::step])

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_path = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_path = output_path / f"coherence_vs_topics_{nmf_method}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    show_plot = False
    # Show plot interactively
    if show_plot:
        plt.show()
    else:
        plt.close()

    return str(plot_path) if plot_path else None


def plot_coherence_comparison(
    results_dict: dict,
    output_dir: str,
    save_plot: bool = True,
    show_plot: bool = False,
) -> Optional[str]:
    """
    Plot coherence scores for multiple NMF methods on the same graph.

    Useful for comparing different NMF variants (nmf, pnmf, nmtf) to see
    which performs best across topic counts.

    Args:
        results_dict: Dictionary mapping method names to OptimizationResult objects.
        output_dir: Directory to save the plot.
        save_plot: Whether to save plot to file.
        show_plot: Whether to display interactive plot.

    Returns:
        Path to saved plot file (if saved), None otherwise.
    """
    if not results_dict:
        return None

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["b", "g", "r", "c", "m", "y"]
    markers = ["o", "s", "^", "D", "v", "p"]

    for i, (method, result) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(
            result.topic_counts,
            result.coherence_scores,
            f"{color}-{marker}",
            linewidth=2,
            markersize=6,
            label=f"{method.upper()} (best: {result.optimal_topic_count} topics)",
        )

        # Mark optimal point for each method
        optimal_idx = int(np.argmax(result.coherence_scores))
        ax.plot(
            result.topic_counts[optimal_idx],
            result.coherence_scores[optimal_idx],
            f"{color}*",
            markersize=15,
            zorder=5,
        )

    # Configure axes
    ax.set_xlabel("Number of Topics", fontsize=12)
    ax.set_ylabel("Gensim C_V Coherence Score", fontsize=12)
    ax.set_title(
        "Topic Count Optimization - Method Comparison",
        fontsize=14,
        fontweight="bold",
    )

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_path = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_path = output_path / "coherence_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    show_plot = False
    # Show plot interactively
    if show_plot:
        plt.show()
    else:
        plt.close()

    return str(plot_path) if plot_path else None
