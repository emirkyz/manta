"""
Temporal topic distribution visualization.

This module provides functions for visualizing how topics evolve over time.
"""

from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ...utils.analysis import get_dominant_topics


def gen_temporal_topic_dist(
    W: np.ndarray,
    datetime_series: pd.Series,
    output_dir: Union[str, Path],
    table_name: str,
    time_grouping: str = 'year',
    normalize: bool = True,
    min_score: float = 0.0,
    plot_type: str = 'stacked_area',
    figsize: tuple = (14, 8)
) -> tuple:
    """
    Generate temporal distribution plot showing how topics evolve over time.

    Args:
        W (numpy.ndarray): Document-topic matrix where rows are documents and columns are topics.
        datetime_series (pd.Series): Series containing datetime information for each document.
        output_dir (str|Path): Directory to save the plots.
        table_name (str): Name of the table/dataset.
        time_grouping (str): How to group time periods. Options: 'year', 'quarter', 'month', 'week'.
        normalize (bool): If True, normalize counts to show proportions instead of raw counts.
        min_score (float): Minimum topic score threshold for dominant topic assignment.
        plot_type (str): Type of plot. Options: 'stacked_area', 'line', 'stacked_bar', 'heatmap'.
        figsize (tuple): Figure size for the plot.

    Returns:
        tuple: (matplotlib figure, temporal distribution dataframe)
    """
    print(f"Generating temporal topic distribution (grouped by {time_grouping})...")

    # Validate inputs
    if len(W) != len(datetime_series):
        raise ValueError(f"W matrix rows ({len(W)}) must match datetime_series length ({len(datetime_series)})")

    # Convert datetime_series to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(datetime_series):
        datetime_series = pd.to_datetime(datetime_series)

    # Get dominant topics
    dominant_topics = get_dominant_topics(W, min_score=min_score)

    # Create DataFrame with topic assignments and datetime
    df = pd.DataFrame({
        'datetime': datetime_series.values,
        'topic': dominant_topics
    })

    # Filter out invalid topics (-1)
    valid_mask = df['topic'] != -1
    df = df[valid_mask]
    excluded_count = (~valid_mask).sum()

    if excluded_count > 0:
        print(f"Excluded {excluded_count} documents with insufficient topic scores")

    if len(df) == 0:
        raise ValueError("No valid documents after filtering. Check min_score parameter.")

    # Group by time period
    if time_grouping == 'year':
        df['period'] = df['datetime'].dt.year
    elif time_grouping == 'quarter':
        df['period'] = df['datetime'].dt.to_period('Q').astype(str)
    elif time_grouping == 'month':
        df['period'] = df['datetime'].dt.to_period('M').astype(str)
    elif time_grouping == 'week':
        df['period'] = df['datetime'].dt.to_period('W').astype(str)
    else:
        raise ValueError(f"Invalid time_grouping: {time_grouping}. Use 'year', 'quarter', 'month', or 'week'")

    # Count topics per period
    temporal_dist = df.groupby(['period', 'topic']).size().unstack(fill_value=0)

    # Ensure all topics are represented
    n_topics = W.shape[1]
    for topic_idx in range(n_topics):
        if topic_idx not in temporal_dist.columns:
            temporal_dist[topic_idx] = 0

    # Sort columns by topic index
    temporal_dist = temporal_dist[sorted(temporal_dist.columns)]

    # Rename columns to start from Topic 1
    temporal_dist.columns = [f'Topic {i+1}' for i in temporal_dist.columns]

    # Normalize if requested
    if normalize:
        temporal_dist = temporal_dist.div(temporal_dist.sum(axis=1), axis=0) * 100

    print(f"\nTemporal distribution summary:")
    print(f"Time periods: {len(temporal_dist)}")
    print(f"Topics: {len(temporal_dist.columns)}")
    print(f"Total documents: {len(df)}")

    # Create plot based on type
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'stacked_area':
        temporal_dist.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        ylabel = 'Proportion of Documents (%)' if normalize else 'Number of Documents'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        title = f'Topic Distribution Over Time (Stacked Area)'

    elif plot_type == 'line':
        temporal_dist.plot(kind='line', ax=ax, marker='o', linewidth=2)
        ylabel = 'Proportion of Documents (%)' if normalize else 'Number of Documents'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        title = f'Topic Distribution Over Time (Line Plot)'

    elif plot_type == 'stacked_bar':
        temporal_dist.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        ylabel = 'Proportion of Documents (%)' if normalize else 'Number of Documents'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        title = f'Topic Distribution Over Time (Stacked Bar)'
        plt.xticks(rotation=45, ha='right')

    elif plot_type == 'heatmap':
        plt.close(fig)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(temporal_dist.T, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Proportion (%)' if normalize else 'Count'})
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        ax.set_ylabel('Topics', fontsize=12)
        title = f'Topic Distribution Heatmap Over Time'
        plt.xticks(rotation=45, ha='right')

    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Use 'stacked_area', 'line', 'stacked_bar', or 'heatmap'")

    # Set x-axis limits to prevent padding beyond actual data range
    if plot_type != 'heatmap':
        if time_grouping == 'year':
            # For year grouping, use actual min/max years as integers
            year_min = int(df["datetime"].min().strftime('%Y'))
            year_max = int(df["datetime"].max().strftime('%Y'))
            ax.set_xlim(year_min, year_max)


        else:
            # For other groupings, set limits based on positions
            ax.set_xlim(-0.5, len(temporal_dist.index) - 0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Create output directory
    output_dir_path = Path(output_dir)
    if output_dir_path.name == table_name:
        table_output_dir = output_dir_path
    else:
        table_output_dir = output_dir_path / table_name
    table_output_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plot_filename = f"{table_name}_temporal_topic_dist_{time_grouping}_{plot_type}.png"
    plot_path = table_output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTemporal topic distribution plot saved to: {plot_path}")

    # Also save the data as CSV
    csv_path = table_output_dir / f"{table_name}_temporal_topic_dist_{time_grouping}.csv"
    temporal_dist.to_csv(csv_path)
    print(f"Temporal distribution data saved to: {csv_path}")

    return fig, temporal_dist


def gen_multi_temporal_plots(
    W: np.ndarray,
    datetime_series: pd.Series,
    output_dir: Union[str, Path],
    table_name: str,
    time_grouping: str = 'year',
    min_score: float = 0.0
) -> List[tuple]:
    """
    Generate multiple temporal distribution plots with different visualization types.

    Args:
        W (numpy.ndarray): Document-topic matrix.
        datetime_series (pd.Series): Series containing datetime information.
        output_dir (str|Path): Directory to save the plots.
        table_name (str): Name of the table/dataset.
        time_grouping (str): How to group time periods.
        min_score (float): Minimum topic score threshold.

    Returns:
        list: List of (figure, dataframe) tuples for each plot type.
    """
    plot_types = ['stacked_area', 'line', 'heatmap']
    results = []

    for plot_type in plot_types:
        print(f"\n{'='*60}")
        print(f"Generating {plot_type} plot...")
        print(f"{'='*60}")

        result = gen_temporal_topic_dist(
            W=W,
            datetime_series=datetime_series,
            output_dir=output_dir,
            table_name=table_name,
            time_grouping=time_grouping,
            normalize=True,
            min_score=min_score,
            plot_type=plot_type
        )
        results.append(result)

    return results