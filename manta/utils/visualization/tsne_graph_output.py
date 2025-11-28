from numpy.distutils.misc_util import colour_text
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple
from ...utils.analysis import get_dominant_topics


def _sort_matrices(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine which word-cluster (H row) best corresponds to which doc-cluster (W column).

    For each column in S (word-cluster), find the row (doc-cluster) with maximum value.
    Returns pairs sorted by coupling strength (descending).

    Args:
        s: S matrix from NMTF (k x k) - coupling between doc-clusters and word-clusters

    Returns:
        ind: Array of (word_cluster_id, doc_cluster_id) tuples, sorted by max coupling
        max_values: Corresponding maximum coupling values
    """
    ind = []
    max_values = []

    for i in range(s.shape[1]):  # For each word-cluster column
        col = s[:, i]
        max_ind = np.argmax(col)  # Find best doc-cluster (row with max value)
        max_values.append(col[max_ind])
        ind.append((i, max_ind))

    ind_sorted = np.argsort(max_values)[::-1]
    ind = np.array(ind)[ind_sorted]
    max_values = np.array(max_values)[ind_sorted]

    return ind, max_values


def _reorder_W_by_pairing(W: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """
    Reorder W matrix columns based on topic pairing from S matrix.

    This ensures that W columns are reordered to match the topic ordering
    determined by the S matrix pairing (sorted by coupling strength).

    Args:
        W: Document-topic matrix (n_docs, n_topics)
        s_matrix: S matrix from NMTF (k x k)

    Returns:
        W_reordered: W matrix with columns reordered by topic pairing
    """
    ind, _ = _sort_matrices(s_matrix)
    n_topics = W.shape[1]
    W_reordered = np.zeros_like(W)

    for new_idx, (word_cluster_id, doc_cluster_id) in enumerate(ind):
        if new_idx < n_topics:
            W_reordered[:, new_idx] = W[:, doc_cluster_id]

    return W_reordered


def tsne_graph_output(w: np.ndarray, h: np.ndarray,
                      s_matrix: Optional[np.ndarray] = None,
                      output_dir: Optional[Union[str, Path]] = None,
                      table_name: str = "tsne_plot",
                      time_data: Optional[pd.Series] = None,
                      time_ranges: Optional[List] = None,
                      cumulative: bool = True,
                      time_column_name: str = "time",
                      outlier_percentile: float = 0.95) -> Optional[str]:
    """
    Create beautiful t-SNE visualizations for document-topic analysis.

    This function generates modern, aesthetically pleasing t-SNE plots that show how documents
    cluster by topic in a 2D space. Supports both static and time-series visualizations.

    Args:
        w: Document-topic matrix (W from NMF/LDA) - shape (n_docs, n_topics)
        h: Topic-word matrix (H from NMF/LDA) - shape (n_topics, n_words)
        s_matrix: Optional S matrix from NMTF (k×k). When provided, uses topic relationships
                 for enhanced clustering. Expected to be L1 column-normalized for consistent
                 interpretation (each column sums to 1.0).
        output_dir: Directory to save the plot (optional)
        table_name: Base name for the output file and plot title
        time_data: Time/date information for time-series visualization (optional)
        time_ranges: List of time points for time-series subplots (optional)
        cumulative: If True, show cumulative data up to each time point
        time_column_name: Name of time column for display purposes
        outlier_percentile: Fraction of points to keep based on distance from center (default 0.95)

    Returns:
        Path to saved plot file, or None if saving failed

    Features:
        - Modern, clean aesthetic with professional color schemes
        - Adaptive point sizing based on dataset size
        - Intelligent density reduction for large datasets
        - Colored points by topic assignment (no legend)
        - Time-series support with multiple subplot layouts
        - High-resolution output suitable for publications
        - NMTF-aware: Uses normalized S matrix for topic relationship weighting

    Note:
        When used with NMTF, the S matrix is expected to be column-normalized (L1 norm)
        where each column sums to 1.0. This ensures consistent probability-like interpretation
        of topic relationships across visualizations.
    """
    # Input validation with beautiful formatting
    print(f"\n🎨 Starting t-SNE Visualization")
    print(f"📊 Input Data:")
    print(f"  • Documents: {w.shape[0] if w is not None else 'None'}")
    print(f"  • Topics: {h.shape[0] if h is not None else 'None'}")
    if time_data is not None:
        print(f"  • Time-series mode: {len(time_ranges) if time_ranges else 'Auto-detect'} periods")
    print(f"  • Output: {table_name}")

    if w is None or h is None:
        print("⚠️  Error: Invalid input matrices for t-SNE visualization")
        return None

    if w.shape[0] < 2:
        print("⚠️  Error: Need at least 2 documents for t-SNE visualization")
        return None

    print(f"\n🔮 Generating t-SNE embedding for {w.shape[0]:,} documents and {h.shape[0]} topics...")

    # Convert W to dense array only if necessary
    if hasattr(w, 'toarray'):
        # It's a sparse matrix, convert to dense
        w_dense = w.toarray()
        print(f"🔄 Converted sparse matrix to dense: {w.shape} → {w_dense.shape}")
    else:
        # Use np.asarray to avoid copying if already an array
        w_dense = np.asarray(w)
        print(f"✅ Using matrix as-is: {w_dense.shape}")

    # Apply t-SNE to document-topic matrix (W) with optimized parameters
    n_docs = w_dense.shape[0]

    # Adaptive perplexity based on dataset size
    adaptive_perplexity = min(30, max(5, n_docs // 3))

    # Choose method based on dataset size
    method = 'barnes_hut' if n_docs > 1000 else 'exact'

    print(f"⚙️  t-SNE parameters: perplexity={adaptive_perplexity}, method={method}, iterations=300")

    tsne = TSNE(
        random_state=42,
        perplexity=adaptive_perplexity,
        n_iter=300,  # Reduced from default 1000
        learning_rate='auto',
        method=method
    )
    tsne_embedding = tsne.fit_transform(w_dense)
    tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x', 'y'])

    # For NMTF models, reorder W columns based on topic pairing from S matrix
    if s_matrix is not None:
        w_for_topics = _reorder_W_by_pairing(w_dense, s_matrix)
    else:
        w_for_topics = w_dense

    # Get dominant topics, filtering out zero-score documents
    dominant_topics = get_dominant_topics(w_for_topics, min_score=0.0)
    tsne_embedding['hue'] = dominant_topics

    # Filter out documents with no dominant topic (marked as -1)
    valid_mask = tsne_embedding['hue'] != -1
    excluded_count = (~valid_mask).sum()

    if excluded_count > 0:
        print(f"Excluded {excluded_count} documents with all zero topic scores from t-SNE visualization")

    tsne_embedding = tsne_embedding[valid_mask].reset_index(drop=True)

    # Remove outliers to improve visualization scaling
    tsne_embedding = _remove_outliers_percentile(tsne_embedding, percentile=outlier_percentile)

    # Apply density-based point reduction if points are too clustered
    tsne_embedding = _apply_density_based_reduction(tsne_embedding)

    # Check if time-series visualization is requested
    if time_data is not None and time_ranges is not None:
        return _create_time_series_visualization(
            tsne_embedding, time_data, time_ranges, cumulative,
            output_dir, table_name, time_column_name
        )

    # Create the standard single visualization with modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='white', edgecolor='none')

    data = tsne_embedding

    # Calculate adaptive point size and transparency based on dataset size
    n_points = len(data)
    # Improved point size calculation: larger points for small datasets
    if n_points <= 100:
        point_size = 50  # Large points for small datasets
    elif n_points <= 500:
        point_size = 35  # Medium-large points
    elif n_points <= 1000:
        point_size = 25  # Medium points
    else:
        point_size = max(8, 25 - np.log10(max(n_points, 10)) * 4)  # Adaptive for large datasets

    # Improved alpha calculation: higher transparency for small datasets, lower for large ones
    if n_points <= 100:
        alpha = 0.95  # Very visible for small datasets
    elif n_points <= 500:
        alpha = 0.85  # Still very visible for medium-small datasets
    elif n_points <= 1000:
        alpha = 0.75  # Good visibility for medium datasets
    else:
        alpha = max(0.6, 0.9 - (n_points / 5000))  # Adaptive transparency for large datasets

    alpha = 1

    print(f"✨ Visualization settings: {n_points} points, size={point_size:.1f}, alpha={alpha:.2f}")

    # Use maximally distinct colors like digital city maps
    import matplotlib.cm as cm
    unique_topics = sorted(data['hue'].unique())
    n_unique_topics = len(unique_topics)
    
    # Generate distinct colors for topics
    distinct_colors = _generate_distinct_colors(n_unique_topics)
    print(f"🎨 Generated {len(distinct_colors)} maximally distinct colors for topics")
    
    # Create a custom colormap from distinct colors
    from matplotlib.colors import ListedColormap
    colormap = ListedColormap(distinct_colors)

    # Clean scatter plot without outlines for better visual clarity
    scatter = ax.scatter(data['x'], data['y'], s=point_size,c = data["hue"],
                         cmap=colormap,alpha=alpha, edgecolors="black", linewidths=0.05)

    # Add legend for topic IDs
    legend_handles = []
    for idx, topic_id in enumerate(unique_topics):
        color = distinct_colors[idx] if idx < len(distinct_colors) else (0.5, 0.5, 0.5, 1.0)
        legend_handles.append(mpatches.Patch(color=color, label=f'Topic {topic_id + 1}'))

    # Move legend below graph to prevent overlap with many topics
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=min(6, len(unique_topics)), fontsize=10, framealpha=0.9,
              title='Topics', title_fontsize=11)

    # Set modern title and labels with better typography
    title_text = f'📊 Topic Distribution Visualization\n{table_name.replace("_", " ").title()}'
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25,
                 color='#2E3440', family='sans-serif')
    ax.set_xlabel('t-SNE Component 1', fontsize=13, color='#4C566A', fontweight='medium')
    ax.set_ylabel('t-SNE Component 2', fontsize=13, color='#4C566A', fontweight='medium')

    # Improve grid and background
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#ffffff')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E9F0')
    ax.spines['bottom'].set_color('#E5E9F0')

    # Add subtle tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#4C566A')
    ax.tick_params(axis='both', which='minor', labelsize=8, colors='#4C566A')

    # Format axis values to be more readable
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    plt.tight_layout()

    # Save the plot with high quality settings
    saved_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{table_name}_tsne_visualization.png"
        file_path = output_path / filename

        # High-quality save settings for professional output
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    pad_inches=0.2, format='png',
                    metadata={'Title': f't-SNE Visualization: {table_name}',
                              'Description': 'Generated by MANTA Topic Modeling'})
        saved_path = str(file_path)
        print(f"💾 High-quality plot saved: {saved_path}")

    plt.show()

    # Print beautiful summary statistics with emojis and formatting
    print(f"\n📊 t-SNE Visualization Summary:")
    print(f"📝 Total documents: {len(data):,}")
    print(f"🎨 Number of topics: {len(unique_topics)}")
    print(f"🔍 Topic Distribution:")
    for topic_id in unique_topics:
        topic_count = len(data[data['hue'] == topic_id])
        percentage = (topic_count / len(data)) * 100
        print(f"  • Topic {topic_id + 1}: {topic_count:,} documents ({percentage:.1f}%)")

    return saved_path


def _create_time_series_visualization(tsne_embedding: pd.DataFrame,
                                      time_data: pd.Series, time_ranges: List,
                                      cumulative: bool, output_dir: Optional[Union[str, Path]],
                                      table_name: str, time_column_name: str) -> Optional[str]:
    """
    Create time-series t-SNE visualization with multiple subplots showing evolution over time.

    Args:
        tsne_embedding: DataFrame with x, y coordinates and hue (topic assignments)
        time_data: Series containing time/date information for each document
        time_ranges: List of time points to create subplots for
        cumulative: If True, show "up to time X", if False show "only time X"
        output_dir: Directory to save the plot
        table_name: Base name for output files
        time_column_name: Name of the time column for display

    Returns:
        Path to saved plot or None if failed
    """
    import matplotlib.cm as cm
    from datetime import datetime

    print(f"🕰️ Creating time-series visualization with {len(time_ranges)} periods...")

    # Parse time data if needed
    parsed_time_data = _parse_time_data(time_data)
    if parsed_time_data is None:
        print("⚠️  Could not parse time data, falling back to standard visualization")
        return None

    # Create subplot layout (2x3 for 6 periods, 3x2 for 6 periods, etc.)
    n_periods = len(time_ranges)
    if n_periods <= 4:
        rows, cols = 2, 2
    elif n_periods <= 6:
        rows, cols = 2, 3
    elif n_periods <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4  # Max 12 periods

    # Create figure and subplots
    plt.style.use('ggplot')
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    if n_periods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Use maximally distinct colors for time-series (same as main visualization)
    unique_topics = sorted(tsne_embedding['hue'].unique())
    n_unique_topics = len(unique_topics)
    
    # Generate distinct colors for topics
    distinct_colors = _generate_distinct_colors(n_unique_topics)
    print(f"🎨 Time-series: Generated {len(distinct_colors)} maximally distinct colors")
    
    # Create a custom colormap from distinct colors
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(distinct_colors)

    # Create each time period subplot
    for idx, time_point in enumerate(time_ranges[:n_periods]):  # Limit to available subplots
        ax = axes[idx]

        # Filter data based on time
        if cumulative:
            # Show all data up to this time point
            mask = parsed_time_data <= time_point
            title_prefix = f"Until {time_point}"
        else:
            # Show only data from this specific time period (you might want to define ranges)
            mask = parsed_time_data == time_point
            title_prefix = f"In {time_point}"

        filtered_data = tsne_embedding[mask]

        if len(filtered_data) > 0:
            # Calculate adaptive visualization settings for this subplot
            n_subplot_points = len(filtered_data)
            # Improved subplot point size calculation
            if n_subplot_points <= 50:
                subplot_point_size = 40  # Large points for small subplots
            elif n_subplot_points <= 200:
                subplot_point_size = 30  # Medium-large points
            elif n_subplot_points <= 500:
                subplot_point_size = 20  # Medium points
            else:
                subplot_point_size = max(6, 20 - np.log10(max(n_subplot_points, 10)) * 3)  # Adaptive

            # Improved subplot alpha calculation: higher transparency for small datasets
            if n_subplot_points <= 50:
                subplot_alpha = 0.95  # Very visible for small subplots
            elif n_subplot_points <= 200:
                subplot_alpha = 0.85  # Still very visible for medium-small subplots
            elif n_subplot_points <= 500:
                subplot_alpha = 0.75  # Good visibility for medium subplots
            else:
                subplot_alpha = max(0.5, 1.0 - (n_subplot_points / 3000))  # Adaptive for large subplots

            # Style the subplot
            ax.set_facecolor('#FFFFFF')
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)

            # Create clean scatter plot without outlines for time-series subplots
            scatter = ax.scatter(
                filtered_data['x'], filtered_data['y'], s=subplot_point_size, c=filtered_data['hue'],
                cmap=cmap, alpha=subplot_alpha
            )

            # Add legend for time-series subplots
            legend_handles = []
            for idx, topic_id in enumerate(sorted(filtered_data['hue'].unique())):
                if idx < len(distinct_colors):
                    color = distinct_colors[idx]
                else:
                    color = (0.5, 0.5, 0.5, 1.0)
                legend_handles.append(mpatches.Patch(color=color, label=f'T{int(topic_id) + 1}'))

            if legend_handles:
                ax.legend(handles=legend_handles, loc='best', fontsize=8, framealpha=0.9)

            ax.set_title(f'{title_prefix}\\n({len(filtered_data):,} documents)',
                         fontsize=11, fontweight='bold', color='#2E3440', pad=10)
        else:
            ax.set_title(f'{title_prefix}\\n(No data)', fontsize=11, color='#8FBCBB',
                         style='italic', fontweight='medium')

        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_periods, len(axes)):
        axes[idx].axis('off')

    # Add main title with modern typography
    title_type = "Cumulative" if cumulative else "Period-by-Period"
    main_title = f'🕰️ Topic Evolution Over Time ({title_type})\\n{table_name.replace("_", " ").title()}'
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.96, color='#2E3440')

    plt.tight_layout()

    # Save the plot
    saved_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{table_name}_tsne_time_series.png"
        file_path = output_path / filename

        # High-quality save settings for time-series plot
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    pad_inches=0.2, format='png',
                    metadata={'Title': f'Time-series t-SNE: {table_name}',
                              'Description': 'Generated by MANTA Topic Modeling'})
        saved_path = str(file_path)
        print(f"💾 Time-series plot saved: {saved_path}")

    plt.show()

    # Print beautiful time-series summary
    print(f"\\n🕰️ Time-Series t-SNE Summary:")
    print(f"📅 Time periods: {len(time_ranges)}")
    print(f"🔄 Mode: {'Cumulative' if cumulative else 'Period-by-period'}")
    print(f"📈 Document Distribution:")
    for i, time_point in enumerate(time_ranges):
        if cumulative:
            mask = parsed_time_data <= time_point
            prefix = "Up to"
        else:
            mask = parsed_time_data == time_point
            prefix = "In"
        count = mask.sum()
        print(f"  • {prefix} {time_point}: {count:,} documents")

    return saved_path


def _parse_time_data(time_data: pd.Series) -> Optional[pd.Series]:
    """
    Parse various time formats into a standardized format for filtering.

    Args:
        time_data: Series containing time/date information

    Returns:
        Parsed time series or None if parsing fails
    """
    import pandas as pd
    from datetime import datetime

    if time_data is None or len(time_data) == 0:
        return None

    try:
        # Try different parsing strategies

        # Strategy 1: Already datetime
        if pd.api.types.is_datetime64_any_dtype(time_data):
            return time_data

        # Strategy 2: Numeric years (2020, 2021, etc.)
        if pd.api.types.is_numeric_dtype(time_data):
            # Assume years if values are reasonable (1900-2100)
            min_val, max_val = time_data.min(), time_data.max()
            if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                return pd.to_datetime(time_data, format='%Y', errors='coerce')

        # Strategy 3: String parsing
        if pd.api.types.is_string_dtype(time_data):
            # Try common formats
            for fmt in ['%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    parsed = pd.to_datetime(time_data, format=fmt, errors='coerce')
                    if not parsed.isna().all():
                        return parsed
                except:
                    continue

            # Try general parsing
            try:
                return pd.to_datetime(time_data, errors='coerce')
            except:
                pass

        print(f"⚠️  Could not parse time data of type {time_data.dtype}")
        return None

    except Exception as e:
        print(f"⚠️  Error parsing time data: {e}")
        return None


def _auto_detect_time_ranges(time_data: pd.Series, n_periods: int = 6) -> Optional[List]:
    """
    Automatically detect meaningful time ranges from the data.

    Args:
        time_data: Parsed time series
        n_periods: Number of time periods to create

    Returns:
        List of time points for subplots
    """
    if time_data is None or len(time_data) == 0:
        return None

    try:
        # Remove NaN values
        valid_times = time_data.dropna()
        if len(valid_times) == 0:
            return None

        min_time = valid_times.min()
        max_time = valid_times.max()

        # Create evenly spaced time points
        time_range = pd.date_range(start=min_time, end=max_time, periods=n_periods)

        # Convert to year if the span is multiple years
        time_span_years = (max_time - min_time).days / 365.25
        if time_span_years > 2:
            # Use years
            return [t.year for t in time_range]
        else:
            # Use full dates
            return [t.strftime('%Y-%m-%d') for t in time_range]

    except Exception as e:
        print(f"⚠️  Could not auto-detect time ranges: {e}")
        return None


def _remove_outliers_percentile(tsne_data: pd.DataFrame, percentile: float = 0.95) -> pd.DataFrame:
    """
    Remove outlier points that are far from the main cluster using percentile-based filtering.

    This helps create better-scaled visualizations by removing extreme outliers that would
    otherwise force the axes to scale widely and compress the main cluster.

    Args:
        tsne_data: DataFrame with x, y, hue columns
        percentile: Fraction of points to keep (default 0.95 = keep 95% closest to center)

    Returns:
        Filtered DataFrame with outliers removed
    """
    import numpy as np

    if len(tsne_data) <= 10:  # Don't filter very small datasets
        return tsne_data

    # Calculate center point (using median for robustness)
    center_x = tsne_data['x'].median()
    center_y = tsne_data['y'].median()

    # Calculate Euclidean distance from center for each point
    distances = np.sqrt((tsne_data['x'] - center_x)**2 + (tsne_data['y'] - center_y)**2)

    # Find the distance threshold (keep points within percentile)
    threshold = np.quantile(distances, percentile)

    # Filter points
    mask = distances <= threshold
    filtered_data = tsne_data[mask].reset_index(drop=True)

    # Log the filtering results
    n_removed = len(tsne_data) - len(filtered_data)
    if n_removed > 0:
        print(f"🎯 Removed {n_removed} outlier points (keeping {percentile*100:.1f}% closest to center)")
        print(f"   Distance threshold: {threshold:.2f}")

    return filtered_data


def _apply_representative_sampling(tsne_data: pd.DataFrame, target_size: int = 1500) -> pd.DataFrame:
    """
    Apply representative sampling to reduce point density while preserving data distribution.

    Args:
        tsne_data: DataFrame with x, y, hue columns
        target_size: Target number of points to keep

    Returns:
        Sampled DataFrame with representative points
    """
    from sklearn.cluster import KMeans
    import numpy as np

    if len(tsne_data) <= target_size:
        return tsne_data

    try:
        # Separate by topic to maintain topic representation
        sampled_dfs = []

        for hue in tsne_data['hue'].unique():
            topic_data = tsne_data[tsne_data['hue'] == hue].copy()
            topic_size = len(topic_data)

            # Proportional sampling: each topic gets proportional representation
            topic_target = max(10, int(target_size * (topic_size / len(tsne_data))))
            topic_target = min(topic_target, topic_size)

            if topic_size <= topic_target:
                # Keep all points for small topics
                sampled_dfs.append(topic_data)
            else:
                # Apply clustering-based sampling for large topics
                coords = topic_data[['x', 'y']].values

                # Use fewer clusters for better performance
                n_clusters = min(topic_target // 2, topic_size // 4, 50)

                if n_clusters < 2:
                    # Random sampling for very small targets
                    sampled = topic_data.sample(n=topic_target, random_state=42)
                else:
                    # Cluster-based sampling
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(coords)

                    sampled_indices = []

                    # From each cluster, take cluster center + some random points
                    for cluster_id in range(n_clusters):
                        cluster_mask = clusters == cluster_id
                        cluster_indices = topic_data.index[cluster_mask]

                        if len(cluster_indices) == 0:
                            continue

                        # Take 1-3 points per cluster depending on cluster size
                        points_per_cluster = max(1, min(3, topic_target // n_clusters))

                        if len(cluster_indices) <= points_per_cluster:
                            sampled_indices.extend(cluster_indices)
                        else:
                            # Take the point closest to cluster center + random samples
                            cluster_coords = coords[cluster_mask]
                            center = kmeans.cluster_centers_[cluster_id]

                            # Find closest point to center
                            distances = np.sum((cluster_coords - center) ** 2, axis=1)
                            closest_idx = cluster_indices[np.argmin(distances)]
                            sampled_indices.append(closest_idx)

                            # Add random samples from the cluster
                            remaining_indices = [idx for idx in cluster_indices if idx != closest_idx]
                            if remaining_indices and points_per_cluster > 1:
                                n_random = min(points_per_cluster - 1, len(remaining_indices))
                                random_indices = np.random.choice(remaining_indices, n_random, replace=False)
                                sampled_indices.extend(random_indices)

                    # If we have fewer points than target, add some random ones
                    if len(sampled_indices) < topic_target:
                        remaining_indices = [idx for idx in topic_data.index if idx not in sampled_indices]
                        if remaining_indices:
                            n_additional = min(topic_target - len(sampled_indices), len(remaining_indices))
                            additional_indices = np.random.choice(remaining_indices, n_additional, replace=False)
                            sampled_indices.extend(additional_indices)

                    sampled = topic_data.loc[sampled_indices]

                sampled_dfs.append(sampled)

        # Combine all sampled topics
        result = pd.concat(sampled_dfs, ignore_index=True)
        return result

    except Exception as e:
        print(f"⚠️  Sampling optimization failed: {e}")
        # Fallback to simple random sampling
        return tsne_data.sample(n=min(target_size, len(tsne_data)), random_state=42)


def _apply_density_based_reduction(tsne_data: pd.DataFrame, density_threshold: float = 0.1) -> pd.DataFrame:
    """
    Reduce points only in areas where they are too densely clustered.

    Args:
        tsne_data: DataFrame with x, y, hue columns
        density_threshold: Minimum distance threshold for keeping points

    Returns:
        DataFrame with density-reduced points
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    if len(tsne_data) <= 500:  # Don't reduce small datasets
        return tsne_data

    try:
        print(f"✨ Optimizing visualization with {len(tsne_data):,} points...")

        # Work on each topic separately to maintain topic representation
        reduced_dfs = []

        for hue in tsne_data['hue'].unique():
            topic_data = tsne_data[tsne_data['hue'] == hue].copy()

            if len(topic_data) <= 20:  # Keep small topics as-is
                reduced_dfs.append(topic_data)
                continue

            # Use k-nearest neighbors to find dense areas
            coords = topic_data[['x', 'y']].values

            # Adaptive k based on topic size
            k = min(10, max(3, len(topic_data) // 50))

            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            # Calculate density score (average distance to k nearest neighbors)
            density_scores = distances[:, 1:].mean(axis=1)  # Skip self (index 0)

            # Keep points that are either:
            # 1. In low-density areas (unique/isolated points)
            # 2. Representative of high-density areas
            median_density = np.median(density_scores)

            keep_indices = []

            # Always keep low-density points (isolated/unique points)
            low_density_mask = density_scores >= median_density
            keep_indices.extend(topic_data.index[low_density_mask])

            # For high-density areas, keep representative points
            high_density_indices = topic_data.index[~low_density_mask]
            if len(high_density_indices) > 0:
                high_density_coords = coords[~low_density_mask]

                # Use clustering to find representatives in dense areas
                from sklearn.cluster import KMeans
                n_representatives = max(10, len(high_density_indices) // 10)
                n_representatives = min(n_representatives, len(high_density_indices))

                if n_representatives >= 2 and len(high_density_coords) >= n_representatives:
                    kmeans = KMeans(n_clusters=n_representatives, random_state=42, n_init=5)
                    clusters = kmeans.fit_predict(high_density_coords)

                    # Keep points closest to cluster centers
                    for cluster_id in range(n_representatives):
                        cluster_mask = clusters == cluster_id
                        if not cluster_mask.any():
                            continue

                        cluster_coords = high_density_coords[cluster_mask]
                        center = kmeans.cluster_centers_[cluster_id]

                        # Find closest point to center
                        distances_to_center = np.sum((cluster_coords - center) ** 2, axis=1)
                        closest_local_idx = np.argmin(distances_to_center)

                        # Convert back to global index
                        cluster_indices = high_density_indices[cluster_mask]
                        closest_global_idx = cluster_indices[closest_local_idx]
                        keep_indices.append(closest_global_idx)
                else:
                    # Fallback: keep some random points from high-density area
                    n_keep = max(5, len(high_density_indices) // 5)
                    keep_indices.extend(np.random.choice(high_density_indices,
                                                         min(n_keep, len(high_density_indices)),
                                                         replace=False))

            reduced_topic = topic_data.loc[keep_indices]
            reduced_dfs.append(reduced_topic)

            reduction_ratio = len(reduced_topic) / len(topic_data)
            print(f"  • Topic {hue}: {len(topic_data):,} → {len(reduced_topic):,} points ({reduction_ratio:.1%})")

        result = pd.concat(reduced_dfs, ignore_index=True)
        total_reduction = len(result) / len(tsne_data)
        print(f"✅ Optimization complete: {len(tsne_data):,} → {len(result):,} points ({total_reduction:.1%} retained)")

        return result

    except Exception as e:
        print(f"⚠️  Warning: Optimization failed, using all points: {e}")
        return tsne_data


def _add_topic_background_regions(ax, data: pd.DataFrame, topics: list):
    """
    Add unified colored background regions for topic clusters using alpha shapes or convex hulls.
    Creates unified backgrounds when points accumulate in areas, not individual point backgrounds.

    Args:
        ax: Matplotlib axis object
        data: DataFrame with x, y, hue columns
        topics: List of topic labels
    """
    try:
        from scipy.spatial import ConvexHull
        import matplotlib.cm as cm
        import matplotlib.patches as patches
        from sklearn.cluster import DBSCAN
        import numpy as np

        # Use the same distinct colors as the scatter plot
        unique_topics = sorted(data['hue'].unique())
        n_unique_topics = len(unique_topics)
        
        # Generate distinct colors for background regions (same as main plot)
        distinct_colors = _generate_distinct_colors(n_unique_topics)
        
        print(f"Creating unified background regions for {len(unique_topics)} topics...")

        for i, topic_id in enumerate(unique_topics):
            if topic_id >= len(topics):
                continue

            topic_data = data[data['hue'] == topic_id]

            if len(topic_data) < 3:  # Need at least 3 points for a region
                continue

            try:
                # Get coordinates for this topic
                coords = topic_data[['x', 'y']].values

                # Get distinct color for this topic (same as scatter plot)
                if i < len(distinct_colors):
                    color = distinct_colors[i]
                else:
                    color = (0.5, 0.5, 0.5, 1.0)  # Fallback gray

                # Use DBSCAN to find dense clusters within each topic
                # This creates unified backgrounds only where points actually cluster
                eps = np.std(coords) * 0.3  # Adaptive epsilon based on data spread
                min_samples = max(2, len(coords) // 10)  # At least 2, but scale with data

                if len(coords) >= min_samples:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(coords)

                    # Create background regions for each dense cluster
                    for cluster_id in set(cluster_labels):
                        if cluster_id == -1:  # Skip noise points
                            continue

                        cluster_mask = cluster_labels == cluster_id
                        cluster_coords = coords[cluster_mask]

                        if len(cluster_coords) >= 3:  # Need at least 3 points for hull
                            try:
                                # Create convex hull for this cluster
                                hull = ConvexHull(cluster_coords)
                                hull_points = cluster_coords[hull.vertices]

                                # Create unified background polygon with subtle transparency
                                polygon = patches.Polygon(hull_points, alpha=0.15, facecolor=color,
                                                          edgecolor=color, linewidth=0.8, linestyle='-')
                                ax.add_patch(polygon)

                            except Exception as e:
                                # Fallback: create circle around cluster center
                                center = np.mean(cluster_coords, axis=0)
                                radius = np.std(cluster_coords, axis=0).mean() * 1.5
                                circle = patches.Circle(center, radius, alpha=0.15, facecolor=color,
                                                        edgecolor=color, linewidth=0.8, linestyle='-')
                                ax.add_patch(circle)
                else:
                    # For small topic groups, create a single unified region if points are close
                    if len(coords) >= 3:
                        # Check if points are clustered (not spread out)
                        spread = np.std(coords, axis=0).mean()
                        if spread < np.std(data[['x', 'y']].values, axis=0).mean():  # More clustered than average
                            try:
                                hull = ConvexHull(coords)
                                hull_points = coords[hull.vertices]

                                polygon = patches.Polygon(hull_points, alpha=0.15, facecolor=color,
                                                          edgecolor=color, linewidth=0.8, linestyle='-')
                                ax.add_patch(polygon)
                            except:
                                pass  # Skip if hull creation fails

            except Exception as e:
                print(f"Warning: Could not create unified region for topic {topic_id}: {e}")
                continue

    except ImportError:
        print("Warning: scipy or sklearn not available for unified background regions")
    except Exception as e:
        print(f"Warning: Could not create unified background regions: {e}")


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
    import numpy as np
    
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
    import numpy as np
    
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
            s = np.random.uniform(0.4, 0.8)  # High saturation for distinction
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


