#!/usr/bin/env python3
"""
Custom Temporal Topic Distribution Line Plot Generator.

This script regenerates the temporal topic distribution line plot with custom
topic names instead of the default "Topic 1", "Topic 2", etc.

Usage:
    1. Edit the TOPIC_NAMES dictionary at the bottom of this script
    2. Set the CSV_PATH to your temporal distribution CSV
    3. Run: python tools/custom_temporal_plot.py
"""

import colorsys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _generate_distinct_colors(n_topics: int) -> list:
    """
    Generate maximally distinct colors for topics.

    Uses predefined high-contrast palettes for common topic counts
    and algorithmic generation for larger numbers.

    Args:
        n_topics: Number of distinct colors needed

    Returns:
        List of RGB tuples with maximally distinct colors
    """
    import matplotlib.colors as mcolors

    distinct_palettes = {
        2: ["#E31A1C", "#1F78B4"],
        3: ["#E31A1C", "#33A02C", "#1F78B4"],
        4: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4"],
        5: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A"],
        6: ["#E31A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#FB9A99"],
        7: [
            "#E31A1C",
            "#FF7F00",
            "#33A02C",
            "#1F78B4",
            "#6A3D9A",
            "#FB9A99",
            "#B15928",
        ],
        8: [
            "#E31A1C",
            "#FF7F00",
            "#33A02C",
            "#1F78B4",
            "#6A3D9A",
            "#FB9A99",
            "#B15928",
            "#FDBF6F",
        ],
        9: [
            "#E31A1C",
            "#FF7F00",
            "#33A02C",
            "#1F78B4",
            "#6A3D9A",
            "#FB9A99",
            "#B15928",
            "#FDBF6F",
            "#CAB2D6",
        ],
        10: [
            "#E31A1C",
            "#FF7F00",
            "#33A02C",
            "#1F78B4",
            "#6A3D9A",
            "#FB9A99",
            "#B15928",
            "#FDBF6F",
            "#CAB2D6",
            "#FFFF99",
        ],
    }

    if n_topics <= 10 and n_topics in distinct_palettes:
        colors = distinct_palettes[n_topics]
        return [mcolors.hex2color(color) for color in colors]

    if n_topics <= 20:
        base_colors = plt.cm.tab20(np.arange(20))
        optimized_order = [
            0,
            10,
            2,
            12,
            4,
            14,
            6,
            16,
            8,
            18,
            1,
            11,
            3,
            13,
            5,
            15,
            7,
            17,
            9,
            19,
        ]
        reordered_colors = [base_colors[i] for i in optimized_order[:n_topics]]
        return [(r, g, b, a) for r, g, b, a in reordered_colors]

    # For very large numbers, use greedy color selection
    colors = []
    colors.append((0.8, 0.2, 0.2, 1.0))

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

        if best_color:
            colors.append(best_color)
        else:
            h = i / n_topics
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.7)
            colors.append((r, g, b, 1.0))

    return colors


def create_temporal_line_plot(
    csv_path: str,
    topic_names: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    dpi: int = 500,
    title: str = "Topic Distribution Over Time",
) -> None:
    """
    Create a temporal topic distribution line plot with custom topic names.

    Args:
        csv_path: Path to the temporal topic distribution CSV file.
        topic_names: Dictionary mapping topic numbers to custom names.
                     Example: {1: "Healthcare Policy", 2: "Medical Education", ...}
        output_path: Path to save the output PNG. If None, saves next to CSV.
        figsize: Figure size as (width, height) tuple.
        dpi: Resolution of output image.
        title: Plot title.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

    # Identify topic columns (all columns except 'period')
    topic_columns = [col for col in df.columns if col != "period"]
    n_topics = len(topic_columns)
    print(f"Found {n_topics} topics: {topic_columns}")

    # Rename columns based on topic_names dictionary
    rename_map = {}
    for col in topic_columns:
        # Extract topic number from column name (e.g., "Topic 1" -> 1)
        if col.startswith("Topic "):
            try:
                topic_num = int(col.replace("Topic ", ""))
                if topic_num in topic_names:
                    rename_map[col] = topic_names[topic_num]
                else:
                    rename_map[col] = col  # Keep original if not in mapping
            except ValueError:
                rename_map[col] = col
        else:
            rename_map[col] = col

    df = df.rename(columns=rename_map)
    print(f"Renamed columns: {list(df.columns)}")

    # Get the new topic column names (after renaming)
    new_topic_columns = [rename_map.get(col, col) for col in topic_columns]

    # Generate distinct colors
    distinct_colors = _generate_distinct_colors(n_topics)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Set period as index for plotting
    df_plot = df.set_index("period")

    # Plot each topic as a line
    for i, col in enumerate(new_topic_columns):
        ax.plot(
            df_plot.index,
            df_plot[col],
            marker="o",
            linewidth=2,
            markersize=3,
            alpha=0.8,
            color=distinct_colors[i],
            label=col,
        )

    # Styling
    ax.set_ylabel("Number of Documents", fontsize=12)
    ax.set_xlabel("Time (Year)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    # Set x-axis ticks to show all years
    ax.set_xticks(df_plot.index)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right")

    # Legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, n_topics),
        fontsize=9,
        framealpha=0.9,
    )
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_custom_names.png"
    else:
        output_path = Path(output_path)

    # Save plot
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
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

    # Path to the temporal topic distribution CSV
    CSV_PATH = "/Users/emirkarayagiz/Work/nmf-standalone/results/bildiri/TopicAnalysis/Output/medicaleducation_pnmf_bpe_10/medicaleducation_pnmf_bpe_10_temporal_topic_dist_year.csv"

    # Optional: Custom output path (set to None to auto-generate)
    OUTPUT_PATH = None

    # Optional: Custom plot title
    TITLE = "Topic Distribution Over Time"

    # =========================================================================
    # END EDITABLE SECTION
    # =========================================================================

    create_temporal_line_plot(
        csv_path=CSV_PATH,
        topic_names=TOPIC_NAMES,
        output_path=OUTPUT_PATH,
        title=TITLE,
    )
