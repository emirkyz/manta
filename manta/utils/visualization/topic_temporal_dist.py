"""
Temporal topic distribution visualization.

This module provides functions for visualizing how topics evolve over time.
"""

from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from ...utils.analysis import get_dominant_topics


def normalize_W_matrix(W:np.ndarray) -> np.ndarray:
    """
    Normalize W matrix so each document (row) sums to 1.

    Args:
        W: Document-topic matrix where rows are documents and columns are topics

    Returns:
        Normalized W matrix where each row sums to 1
    """
    W_normalized = W / W.sum(axis=1, keepdims=True)
    return W_normalized


def gen_temporal_topic_dist(
    W: np.ndarray,
    s_matrix: Optional[np.ndarray] = None,
    datetime_series: pd.Series = None,
    output_dir: Union[str, Path] = None,
    table_name: str = "temporal_dist",
    time_grouping: str = 'year',
    normalize: bool = True,
    min_score: float = 0.0,
    plot_type: str = 'stacked_area',
    figsize: tuple = (14, 8),
    smooth: bool = False,
    use_weighted: bool = False,
    use_mm_yyyy_format: bool = False
) -> tuple:
    """
    Generate temporal distribution plot showing how topics evolve over time.

    Args:
        W (numpy.ndarray): Document-topic matrix where rows are documents and columns are topics.
        s_matrix (numpy.ndarray, optional): S matrix for NMTF. If provided, W @ s_matrix is used before normalization.
        datetime_series (pd.Series): Series containing datetime information for each document.
        output_dir (str|Path): Directory to save the plots.
        table_name (str): Name of the table/dataset.
        time_grouping (str): How to group time periods. Options: 'year', 'quarter', 'month', 'week'.
        normalize (bool): If True, normalize counts to show proportions instead of raw counts.
        min_score (float): Minimum topic score threshold for dominant topic assignment (only used when use_weighted=False).
        plot_type (str): Type of plot. Options: 'stacked_area', 'line', 'stacked_bar', 'heatmap'.
        figsize (tuple): Figure size for the plot.
        smooth (bool): If True, apply smoothing to the data (cubic interpolation for line plots).
        use_weighted (bool): If True, use normalized topic weights instead of counting dominant topics.
                           Only applies to 'line' and 'stacked_area' plot types. When True, each document
                           contributes its normalized topic distribution to the temporal sum, providing a
                           more nuanced view of topic evolution. When False, uses traditional dominant topic
                           assignment (hard assignment).

    Returns:
        tuple: (matplotlib figure, temporal distribution dataframe)
    """
    print(f"Generating temporal topic distribution (grouped by {time_grouping})...")

    # Validate inputs
    if len(W) != len(datetime_series):
        raise ValueError(f"W matrix rows ({len(W)}) must match datetime_series length ({len(datetime_series)})")
    
    if W.shape[0] == 0:
        raise ValueError("W matrix is empty")
    
    if W.shape[1] == 0:
        raise ValueError("W matrix has no topics")

    # Convert datetime_series to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(datetime_series):
        try:
            datetime_series = pd.to_datetime(datetime_series)
        except Exception as e:
            raise ValueError(f"Failed to convert datetime_series to datetime: {e}")
    
    # Check for valid datetime range
    if datetime_series.isna().all():
        raise ValueError("All datetime values are NaN")
    
    # Remove rows with NaN datetime values
    valid_datetime_mask = ~datetime_series.isna()
    if not valid_datetime_mask.all():
        print(f"Warning: Removing {(~valid_datetime_mask).sum()} documents with invalid datetime values")
        datetime_series = datetime_series[valid_datetime_mask]
        W = W[valid_datetime_mask]

    # Determine if we should use weighted approach
    # Use weighted for line and stacked_area plots when use_weighted=True
    use_weighted_approach = use_weighted and plot_type in ['line', 'stacked_area']

    if use_weighted_approach:
        # Weighted approach: sum normalized topic weights over time
        print(f"Using weighted approach: summing normalized topic weights")

        # Apply S matrix if provided (for NMTF)
        if s_matrix is not None:
            print(f"Applying S matrix transformation before normalization")
            W_effective = W @ s_matrix
        else:
            W_effective = W.copy()

        # Normalize the W matrix so each document (row) sums to 1
        W_normalized = normalize_W_matrix(W_effective)

        # Create DataFrame with datetime and all topic weights
        topic_columns = [f'Topic {i+1}' for i in range(W_normalized.shape[1])]
        df_data = {'datetime': datetime_series.values}
        for i, col_name in enumerate(topic_columns):
            df_data[col_name] = W_normalized[:, i]

        df = pd.DataFrame(df_data)

        # Group by time period
        if time_grouping == 'year':
            df['period'] = df['datetime'].dt.year
        elif time_grouping == 'quarter':
            df['period'] = df['datetime'].dt.to_period('Q').astype(str)
        elif time_grouping == 'month':
            if use_mm_yyyy_format:
                # Format as MM-YYYY for combined year/month columns
                df['period'] = df['datetime'].dt.strftime('%m-%Y')
            else:
                # Default format: YYYY-MM
                df['period'] = df['datetime'].dt.to_period('M').astype(str)
        elif time_grouping == 'week':
            df['period'] = df['datetime'].dt.to_period('W').astype(str)
        else:
            raise ValueError(f"Invalid time_grouping: {time_grouping}. Use 'year', 'quarter', 'month', or 'week'")

        # Sum topic weights by period
        temporal_dist = df.groupby('period')[topic_columns].sum()

    else:
        # Original approach: count documents by dominant topic
        print(f"Using count approach: assigning documents to dominant topics")

        # Get dominant topics
        dominant_topics = get_dominant_topics(W, min_score=min_score, s_matrix=s_matrix)

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
            if use_mm_yyyy_format:
                # Format as MM-YYYY for combined year/month columns
                df['period'] = df['datetime'].dt.strftime('%m-%Y')
            else:
                # Default format: YYYY-MM
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
    
    # Validate temporal distribution
    if temporal_dist.empty:
        raise ValueError("No temporal distribution data generated. Check input data and min_score parameter.")

    # Sort the temporal distribution chronologically
    # For MM-YYYY format, we need to convert back to datetime for proper sorting
    if use_mm_yyyy_format and time_grouping == 'month':
        # Parse MM-YYYY back to datetime for sorting
        period_dates = pd.to_datetime(temporal_dist.index, format='%m-%Y')
        # Sort by the parsed dates
        temporal_dist = temporal_dist.loc[period_dates.sort_values().strftime('%m-%Y')]
    elif time_grouping == 'year':
        # For year, ensure integer sorting
        temporal_dist = temporal_dist.sort_index()
    else:
        # For other formats, rely on pandas default sorting
        temporal_dist = temporal_dist.sort_index()

    # Generate distinct colors for topics (consistent with t-SNE visualization)
    n_topics = W.shape[1]
    distinct_colors = _generate_distinct_colors(n_topics)
    print(f"Generated {len(distinct_colors)} maximally distinct colors for {n_topics} topics")

    # Check if we have sufficient data for meaningful analysis
    total_periods = len(temporal_dist)
    if total_periods < 2:
        print(f"Warning: Only {total_periods} time period(s) found. Temporal analysis may not be meaningful.")
    
    # Check for periods with very few documents
    period_totals = temporal_dist.sum(axis=1)
    sparse_periods = period_totals[period_totals < 3]
    if len(sparse_periods) > 0:
        print(f"Warning: {len(sparse_periods)} periods have fewer than 3 documents. Results may be unstable for these periods: {list(sparse_periods.index)}")

    # Normalize if requested - use different methods based on plot type
    normalization_method = None
    if normalize:
        if plot_type in ['stacked_area', 'stacked_bar']:
            # Use percentage normalization for stacked plots (must be all positive)
            # This shows the relative distribution of topics within each time period
            row_sums = temporal_dist.sum(axis=1)

            # Avoid division by zero (though this should not happen with valid data)
            if (row_sums == 0).any():
                print(f"Warning: {(row_sums == 0).sum()} time periods have zero documents")
                row_sums = row_sums.replace(0, 1)  # Replace zeros with 1 to avoid division errors

            temporal_dist = temporal_dist.div(row_sums, axis=0) * 100
            normalization_method = 'percentage'
            print(f"Normalization applied: percentage (each time period sums to 100%)")

        elif plot_type in ['line', 'heatmap']:
            # Use z-score normalization for line plots and heatmaps (can be negative)
            # This shows how each topic deviates from its mean over time
            temporal_dist_normalized = temporal_dist.copy()

            for col in temporal_dist.columns:
                topic_values = temporal_dist[col].values.astype(float)
                topic_mean = np.mean(topic_values)
                topic_std = np.std(topic_values, ddof=1) if len(topic_values) > 1 else 0.0

                # Use z-score normalization if std > 0
                if topic_std > 1e-10:
                    temporal_dist_normalized[col] = (topic_values - topic_mean) / topic_std
                else:
                    # If no variation, center at 0
                    temporal_dist_normalized[col] = np.zeros_like(topic_values)

            temporal_dist = temporal_dist_normalized
            normalization_method = 'z-score'
            print(f"Normalization applied: z-score (shows deviations from mean)")

    print(f"\nTemporal distribution summary:")
    print(f"Time periods: {len(temporal_dist)}")
    print(f"Topics: {len(temporal_dist.columns)}")
    print(f"Total documents: {len(df)}")

    # Apply smoothing if requested (for line plots only)
    temporal_dist_plot = temporal_dist.copy()
    if smooth and plot_type == 'line':
        if len(temporal_dist) >= 4:  # Need at least 4 points for cubic interpolation
            # Create a finer-grained index for smooth interpolation
            if time_grouping == 'year':
                # For years, interpolate with more points
                original_index = temporal_dist.index.astype(int)
                new_index = np.linspace(original_index.min(), original_index.max(), len(original_index) * 10)

                # Interpolate each column
                temporal_dist_smooth = pd.DataFrame(index=new_index, columns=temporal_dist.columns)
                for col in temporal_dist.columns:
                    temporal_dist_smooth[col] = np.interp(new_index, original_index, temporal_dist[col].values)

                # Apply additional smoothing via cubic interpolation
                temporal_dist_smooth = temporal_dist_smooth.interpolate(method='cubic')
                temporal_dist_plot = temporal_dist_smooth
                print(f"Smoothing applied: cubic interpolation with {len(new_index)} points")
            else:
                # For other groupings, use simpler interpolation
                temporal_dist_plot = temporal_dist.interpolate(method='cubic')
                print(f"Smoothing applied: cubic interpolation")
        else:
            print(f"Warning: Not enough data points ({len(temporal_dist)}) for smoothing. Need at least 4 points.")

    # Create plot based on type
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'stacked_area':
        temporal_dist.plot(kind='area', stacked=True, ax=ax, alpha=0.7, color=distinct_colors)

        # Set label based on normalization method and weighted approach
        if normalize and normalization_method == 'z-score':
            ylabel = 'Topic Strength (Z-Score)'
        elif normalize and normalization_method == 'percentage':
            ylabel = 'Topic Distribution (%)'
        elif use_weighted_approach:
            ylabel = 'Topic Weight Sum'
        else:
            ylabel = 'Number of Documents'

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        weighted_suffix = " [Weighted]" if use_weighted_approach else ""
        title = f'Topic Distribution Over Time (Stacked Area){weighted_suffix}'

    elif plot_type == 'line':
        # Use smoothed data if available, otherwise use original
        if smooth:
            temporal_dist_plot.plot(kind='line', ax=ax, linewidth=2.5, alpha=0.8, color=distinct_colors)
        else:
            temporal_dist.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=6, alpha=0.8, color=distinct_colors)

        # Set label based on normalization method and weighted approach
        if normalize and normalization_method == 'z-score':
            ylabel = 'Topic Strength (Z-Score)'
        elif normalize and normalization_method == 'percentage':
            ylabel = 'Topic Distribution (%)'
        elif use_weighted_approach:
            ylabel = 'Topic Weight Sum'
        else:
            ylabel = 'Number of Documents'

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        smooth_suffix = "  [Smoothed]" if smooth else ""
        weighted_suffix = " [Weighted]" if use_weighted_approach else ""
        title = f'Topic Distribution Over Time (Line Plot){weighted_suffix}{smooth_suffix}'

    elif plot_type == 'stacked_bar':
        temporal_dist.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=distinct_colors)

        # Set label based on normalization method
        if normalize and normalization_method == 'z-score':
            ylabel = 'Topic Strength (Z-Score)'
        elif normalize and normalization_method == 'percentage':
            ylabel = 'Topic Distribution (%)'
        else:
            ylabel = 'Number of Documents'

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        title = f'Topic Distribution Over Time (Stacked Bar)'
        plt.xticks(rotation=45, ha='right')

    elif plot_type == 'heatmap':
        plt.close(fig)
        fig, ax = plt.subplots(figsize=figsize)

        # Choose colormap based on normalization method
        if normalize and normalization_method == 'z-score':
            # Diverging colormap for z-scores (centered at 0)
            sns.heatmap(temporal_dist.T, cmap='RdBu_r', ax=ax, center=0,
                       cbar_kws={'label': 'Topic Strength (Z-Score)'})
        else:
            # Sequential colormap for counts or percentages
            cbar_label = 'Topic Distribution (%)' if (normalize and normalization_method == 'percentage') else 'Count'
            sns.heatmap(temporal_dist.T, cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': cbar_label})

        ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
        ax.set_ylabel('Topics', fontsize=12)
        title = f'Topic Distribution Heatmap Over Time'
        plt.xticks(rotation=45, ha='right')

    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Use 'stacked_area', 'line', 'stacked_bar', or 'heatmap'")

    # Set x-axis limits and ticks to show proper integer years from min to max
    if plot_type != 'heatmap':
        if time_grouping == 'year':
            # For year grouping, use actual min/max years as integers
            year_min = int(df["datetime"].min().strftime('%Y'))
            year_max = int(df["datetime"].max().strftime('%Y'))

            # Create tick positions and labels for all years in range
            all_years = list(range(year_min, year_max + 1))

            # Ensure the plot shows data for all years (fill missing years with 0)
            temporal_dist_reindexed = temporal_dist.reindex(all_years, fill_value=0)

            # Re-plot with complete year range if data was missing years
            if len(temporal_dist_reindexed) != len(temporal_dist):
                ax.clear()
                if plot_type == 'stacked_area':
                    temporal_dist_reindexed.plot(kind='area', stacked=True, ax=ax, alpha=0.7, color=distinct_colors)
                    # For area plots, use actual year values on x-axis
                    ax.set_xlim(year_min - 0.5, year_max + 0.5)
                    ax.set_xticks(all_years)
                    ax.set_xticklabels(all_years)
                elif plot_type == 'line':
                    temporal_dist_reindexed.plot(kind='line', ax=ax, marker='o', linewidth=2, color=distinct_colors)
                    # For line plots, use actual year values on x-axis
                    ax.set_xlim(year_min - 0.5, year_max + 0.5)
                    ax.set_xticks(all_years)
                    ax.set_xticklabels(all_years)
                elif plot_type == 'stacked_bar':
                    temporal_dist_reindexed.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=distinct_colors)
                    # For bar plots, use positional indices with year labels
                    ax.set_xticks(range(len(all_years)))
                    ax.set_xticklabels(all_years, rotation=45, ha='right')

                # Re-apply labels
                if normalize and normalization_method == 'z-score':
                    ylabel = 'Topic Strength (Z-Score)'
                elif normalize and normalization_method == 'percentage':
                    ylabel = 'Topic Distribution (%)'
                elif use_weighted_approach:
                    ylabel = 'Topic Weight Sum'
                else:
                    ylabel = 'Number of Documents'
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
            else:
                # No reindexing needed, but still fix bar plot ticks
                if plot_type == 'stacked_bar':
                    # Bar plots use positional indices
                    ax.set_xticks(range(len(temporal_dist.index)))
                    ax.set_xticklabels(temporal_dist.index, rotation=45, ha='right')
                else:
                    # Area and line plots can use actual year values
                    ax.set_xlim(year_min - 0.5, year_max + 0.5)
                    ax.set_xticks(all_years)
                    ax.set_xticklabels(all_years)
        else:
            # For other groupings (month, quarter, week), configure x-axis properly
            if plot_type == 'stacked_bar':
                # Bar plots use positional indices
                ax.set_xticks(range(len(temporal_dist.index)))
                ax.set_xticklabels(temporal_dist.index, rotation=45, ha='right')
            else:
                # For line and area plots, set limits AND labels for proper month display
                ax.set_xlim(-0.5, len(temporal_dist.index) - 0.5)
                ax.set_xticks(range(len(temporal_dist.index)))
                ax.set_xticklabels(temporal_dist.index, rotation=45, ha='right')
    else:
        # For heatmap, ensure proper year labeling
        if time_grouping == 'year':
            year_min = int(df["datetime"].min().strftime('%Y'))
            year_max = int(df["datetime"].max().strftime('%Y'))
            all_years = list(range(year_min, year_max + 1))
            
            # Reindex temporal_dist to include all years
            temporal_dist_reindexed = temporal_dist.reindex(all_years, fill_value=0)
            
            # Re-create heatmap with complete data
            plt.close(fig)
            fig, ax = plt.subplots(figsize=figsize)

            # Choose colormap based on normalization method
            if normalize and normalization_method == 'z-score':
                sns.heatmap(temporal_dist_reindexed.T, cmap='RdBu_r', ax=ax, center=0,
                           cbar_kws={'label': 'Topic Strength (Z-Score)'})
            else:
                cbar_label = 'Topic Distribution (%)' if (normalize and normalization_method == 'percentage') else 'Count'
                sns.heatmap(temporal_dist_reindexed.T, cmap='YlOrRd', ax=ax,
                           cbar_kws={'label': cbar_label})
            ax.set_xlabel(f'Time ({time_grouping.capitalize()})', fontsize=12)
            ax.set_ylabel('Topics', fontsize=12)
            
            # Set proper x-tick labels
            ax.set_xticks(range(len(all_years)))
            ax.set_xticklabels(all_years, rotation=45, ha='right')

    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend and grid only for non-heatmap plots
    if plot_type != 'heatmap':
        # Move legend below for all plot types to prevent overlap with many topics
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                  ncol=min(6, n_topics), fontsize=10, framealpha=0.9)
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
    s_matrix: Optional[np.ndarray] = None,
    time_grouping: str = 'year',
    min_score: float = 0.0,
    smooth: bool = False,
    use_weighted: bool = False
) -> List[tuple]:
    """
    Generate multiple temporal distribution plots with different visualization types.

    Args:
        W (numpy.ndarray): Document-topic matrix.
        datetime_series (pd.Series): Series containing datetime information.
        output_dir (str|Path): Directory to save the plots.
        table_name (str): Name of the table/dataset.
        s_matrix (numpy.ndarray, optional): S matrix for NMTF.
        time_grouping (str): How to group time periods.
        min_score (float): Minimum topic score threshold.
        smooth (bool): If True, apply smoothing to line plots.
        use_weighted (bool): If True, use normalized topic weights instead of counting.

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
            s_matrix=s_matrix,
            datetime_series=datetime_series,
            output_dir=output_dir,
            table_name=table_name,
            time_grouping=time_grouping,
            normalize=True,
            min_score=min_score,
            plot_type=plot_type,
            smooth=smooth,
            use_weighted=use_weighted
        )
        results.append(result)

    return results


def _generate_distinct_colors(n_topics: int) -> List[tuple]:
    """
    Generate maximally distinct colors for topics, similar to digital city maps.

    Uses a combination of predefined high-contrast palettes and perceptual color spacing
    to ensure adjacent topics have visually distinct colors.

    Args:
        n_topics: Number of distinct colors needed

    Returns:
        List of RGB tuples with maximally distinct colors
    """
    import matplotlib.colors as mcolors

    # Predefined high-contrast color palettes for common topic counts
    # These are carefully chosen to be maximally distinct
    distinct_palettes = {
        2: ['#E31A1C', '#1F78B4'],  # Red, Blue
        3: ['#E31A1C', '#33A02C', '#1F78B4'],  # Red, Green, Blue
        4: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4'],  # Red, Orange, Green, Blue
        5: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A'],  # + Purple
        6: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99'],  # + Light Pink
        7: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928'],  # + Brown
        8: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F'],  # + Light Orange
        9: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F', '#CAB2D6'],  # + Light Purple
        10: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F', '#CAB2D6', '#FFFF99'],  # + Yellow
    }

    if n_topics <= 10 and n_topics in distinct_palettes:
        # Use predefined palette for optimal distinction
        colors = distinct_palettes[n_topics]
        return [mcolors.hex2color(color) for color in colors]

    # For larger numbers of topics, use a combination approach
    if n_topics <= 20:
        # Use tab20 colormap but with optimized ordering for maximum distinction
        import matplotlib.pyplot as plt
        base_colors = plt.cm.tab20(np.arange(20))

        # Reorder colors to maximize distinction between adjacent indices
        # Interleave light and dark colors, separate similar hues
        optimized_order = [0, 10, 2, 12, 4, 14, 6, 16, 8, 18, 1, 11, 3, 13, 5, 15, 7, 17, 9, 19]
        reordered_colors = [base_colors[i] for i in optimized_order[:n_topics]]
        return [(r, g, b, a) for r, g, b, a in reordered_colors]

    # For very large numbers of topics, use greedy color selection
    return _generate_greedy_distinct_colors(n_topics)


def _generate_greedy_distinct_colors(n_topics: int) -> List[tuple]:
    """
    Generate colors using a greedy algorithm that maximizes perceptual distance.

    Args:
        n_topics: Number of colors needed

    Returns:
        List of RGB tuples with maximally distinct colors
    """
    import colorsys

    if n_topics <= 1:
        return [(0.8, 0.2, 0.2, 1.0)]  # Default red

    colors = []

    # Start with a high-contrast base color
    colors.append((0.8, 0.2, 0.2, 1.0))  # Red

    # For each additional color, find the one with maximum distance from existing colors
    for i in range(1, n_topics):
        best_color = None
        best_min_distance = 0

        # Try 100 candidate colors
        for _ in range(100):
            # Generate candidate color in HSV space for better perceptual distribution
            h = np.random.uniform(0, 1)
            s = np.random.uniform(0.6, 1.0)  # High saturation for distinction
            v = np.random.uniform(0.4, 0.9)  # Avoid very dark or very light

            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            candidate = (r, g, b, 1.0)

            # Calculate minimum distance to existing colors
            min_distance = min(_color_distance(candidate, existing) for existing in colors)

            # Keep the candidate with the largest minimum distance
            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_color = candidate

        if best_color:
            colors.append(best_color)
        else:
            # Fallback: use HSV with even spacing
            h = i / n_topics
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.7)
            colors.append((r, g, b, 1.0))

    return colors


def _color_distance(color1: tuple, color2: tuple) -> float:
    """
    Calculate perceptual distance between two RGB colors.

    Uses a simple but effective Euclidean distance in RGB space.
    For better results, could be upgraded to LAB color space.

    Args:
        color1: RGB(A) tuple
        color2: RGB(A) tuple

    Returns:
        Float distance value
    """
    r1, g1, b1 = color1[:3]
    r2, g2, b2 = color2[:3]

    # Simple RGB distance (could be improved with LAB color space)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5