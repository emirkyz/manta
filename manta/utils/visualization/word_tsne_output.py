from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any
from ..console.console_manager import ConsoleManager, get_console


def word_tsne_visualization(
    h: np.ndarray,
    vocab: Optional[Union[List[str], dict]] = None,
    tokenizer: Optional[Any] = None,
    s_matrix: Optional[np.ndarray] = None,
    output_dir: Optional[Union[str, Path]] = None,
    table_name: str = "word_tsne",
    top_words_per_topic: int = 50,
    perplexity: int = 30,
    outlier_percentile: float = 0.95,
    console: Optional[ConsoleManager] = None
) -> Optional[str]:
    """
    Create t-SNE visualization for words in topic space.

    This function generates a 2D t-SNE visualization showing how words cluster by topic.
    Words are extracted from the H matrix (topic-word matrix) and projected into 2D space,
    colored by their dominant topic assignment.

    Args:
        h: Topic-word matrix (H from NMF/LDA) - shape (n_topics, n_words)
        vocab: Vocabulary mapping (list of words or dict mapping indices to words)
        tokenizer: Alternative to vocab for token ID mapping (used in Turkish/BPE)
        s_matrix: Optional S matrix from NMTF (k×k). DEPRECATED: Kept for backwards compatibility
        output_dir: Directory to save the plot (optional)
        table_name: Base name for the output file and plot title
        top_words_per_topic: Number of top words to extract per topic (default 50)
        perplexity: t-SNE perplexity parameter (default 30)
        outlier_percentile: Fraction of points to keep based on distance from center (default 0.95)
        console: ConsoleManager instance for logging

    Returns:
        Path to saved plot file, or None if saving failed

    Features:
        - Extracts top N words per topic based on H matrix scores
        - Projects words into 2D space using t-SNE
        - Colors words by dominant topic assignment
        - Adaptive point sizing based on dataset size
        - Professional styling matching document t-SNE visualization
        - High-resolution output suitable for publications
    """
    # Input validation with console output
    _console = console or get_console()
    _console.print_debug("Starting Word t-SNE Visualization", tag="VISUALIZATION")
    _console.print_debug(f"Input Data: Topics={h.shape[0] if h is not None else 'None'}, Vocabulary Size={h.shape[1] if h is not None else 'None'}", tag="VISUALIZATION")
    _console.print_debug(f"Top words per topic: {top_words_per_topic}", tag="VISUALIZATION")
    _console.print_debug(f"Output: {table_name}", tag="VISUALIZATION")

    if h is None:
        _console.print_warning("Invalid H matrix for word t-SNE visualization", tag="VISUALIZATION")
        return None

    if h.shape[0] < 1:
        _console.print_warning("Need at least 1 topic for word t-SNE visualization", tag="VISUALIZATION")
        return None

    if vocab is None and tokenizer is None:
        _console.print_warning("Need either vocab or tokenizer for word mapping", tag="VISUALIZATION")
        return None

    # Convert H to dense array if sparse
    if hasattr(h, 'toarray'):
        h_dense = h.toarray()
        _console.print_debug(f"Converted sparse H matrix to dense: {h.shape} -> {h_dense.shape}", tag="VISUALIZATION")
    else:
        h_dense = np.asarray(h)
        _console.print_debug(f"Using H matrix as-is: {h_dense.shape}", tag="VISUALIZATION")

    n_topics, n_vocab = h_dense.shape

    # Extract top words per topic
    _console.print_debug(f"Extracting top {top_words_per_topic} words from {n_topics} topics...", tag="VISUALIZATION")
    word_indices, word_topics = _extract_top_words_from_h(h_dense, top_words_per_topic, n_vocab, _console)

    if len(word_indices) < 2:
        _console.print_warning(f"Need at least 2 words for t-SNE visualization, got {len(word_indices)}", tag="VISUALIZATION")
        return None

    _console.print_debug(f"Extracted {len(word_indices)} unique words from {n_topics} topics", tag="VISUALIZATION")

    # Create word embeddings from H matrix (transpose to get words as rows)
    h_transposed = h_dense.T  # Shape: (n_vocab, n_topics)
    word_embeddings = h_transposed[word_indices, :]  # Shape: (n_selected_words, n_topics)

    _console.print_debug(f"Created word embeddings: shape {word_embeddings.shape}", tag="VISUALIZATION")

    # Apply t-SNE to word embeddings
    n_words = word_embeddings.shape[0]

    # Adaptive perplexity based on dataset size
    adaptive_perplexity = min(perplexity, max(5, n_words // 3))

    # Choose method based on dataset size
    method = 'barnes_hut' if n_words > 1000 else 'exact'

    _console.print_debug(f"t-SNE parameters: perplexity={adaptive_perplexity}, method={method}, iterations=300", tag="VISUALIZATION")

    tsne = TSNE(
        random_state=42,
        perplexity=adaptive_perplexity,
        n_iter=300,
        learning_rate='auto',
        method=method
    )

    tsne_embedding = tsne.fit_transform(word_embeddings)
    tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x', 'y'])
    tsne_embedding['hue'] = word_topics

    _console.print_debug(f"t-SNE complete for {n_words} words", tag="VISUALIZATION")

    # Remove outliers to improve visualization scaling
    tsne_embedding = _remove_outliers_percentile(tsne_embedding, percentile=outlier_percentile, console=_console)

    # Apply density-based point reduction if points are too clustered
    if len(tsne_embedding) > 500:
        tsne_embedding = _apply_density_based_reduction(tsne_embedding, console=_console)

    # Create the visualization with modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='white', edgecolor='none')

    data = tsne_embedding

    # Calculate adaptive point size and transparency based on dataset size
    n_points = len(data)
    # Improved point size calculation
    if n_points <= 100:
        point_size = 50  # Large points for small datasets
    elif n_points <= 500:
        point_size = 35  # Medium-large points
    elif n_points <= 1000:
        point_size = 25  # Medium points
    else:
        point_size = max(8, 25 - np.log10(max(n_points, 10)) * 4)
    point_size *= 3

    # Improved alpha calculation
    if n_points <= 100:
        alpha = 0.95
    elif n_points <= 500:
        alpha = 0.85
    elif n_points <= 1000:
        alpha = 0.75
    else:
        alpha = max(0.6, 0.9 - (n_points / 5000))

    alpha = 1

    _console.print_debug(f"Visualization settings: {n_points} points, size={point_size:.1f}, alpha={alpha:.2f}", tag="VISUALIZATION")

    # Use maximally distinct colors
    unique_topics = sorted(data['hue'].unique())
    n_unique_topics = len(unique_topics)

    # Generate distinct colors for topics
    distinct_colors = _generate_distinct_colors(n_unique_topics)
    _console.print_debug(f"Generated {len(distinct_colors)} maximally distinct colors for topics", tag="VISUALIZATION")

    # Create a custom colormap from distinct colors
    from matplotlib.colors import ListedColormap
    colormap = ListedColormap(distinct_colors)

    # Clean scatter plot without outlines for better visual clarity
    scatter = ax.scatter(data['x'], data['y'], s=point_size, c=data["hue"],
                         cmap=colormap, alpha=alpha, edgecolors="black", linewidths=0.05)

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
    title_text = f'Word Distribution in Topic Space\\n{table_name.replace("_", " ").title()}'
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

        filename = f"{table_name}_word_tsne_visualization.png"
        file_path = output_path / filename

        # High-quality save settings for professional output
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    pad_inches=0.2, format='png',
                    metadata={'Title': f'Word t-SNE Visualization: {table_name}',
                              'Description': 'Generated by MANTA Topic Modeling'})
        saved_path = str(file_path)
        _console.print_debug(f"High-quality plot saved: {saved_path}", tag="VISUALIZATION")

    plt.show()

    # Print summary statistics
    _console.print_debug(f"Word t-SNE Visualization Summary: {len(data):,} words, {len(unique_topics)} topics", tag="VISUALIZATION")
    for topic_id in unique_topics:
        topic_count = len(data[data['hue'] == topic_id])
        percentage = (topic_count / len(data)) * 100
        _console.print_debug(f"  Topic {topic_id + 1}: {topic_count:,} words ({percentage:.1f}%)", tag="VISUALIZATION")

    return saved_path


def _extract_top_words_from_h(h: np.ndarray, top_n: int, n_vocab: int,
                               console: ConsoleManager) -> Tuple[List[int], List[int]]:
    """
    Extract top N words per topic from H matrix.

    Args:
        h: Topic-word matrix (n_topics, n_words)
        top_n: Number of top words to extract per topic
        n_vocab: Total vocabulary size
        console: Console manager for logging

    Returns:
        Tuple of (word_indices, word_topic_assignments)
        - word_indices: List of unique word indices
        - word_topic_assignments: List of dominant topic for each word
    """
    n_topics = h.shape[0]

    # Track word indices and their topic assignments
    word_to_topics = {}  # word_idx -> list of (topic_id, score)

    for topic_idx in range(n_topics):
        # Get scores for all words in this topic
        topic_scores = h[topic_idx, :]

        # Get top N word indices (sorted by score, descending)
        top_indices = np.argsort(topic_scores)[::-1][:top_n]

        # Record each word and its association with this topic
        for word_idx in top_indices:
            score = topic_scores[word_idx]
            if word_idx not in word_to_topics:
                word_to_topics[word_idx] = []
            word_to_topics[word_idx].append((topic_idx, score))

    # For each word, assign it to the topic where it has the highest score
    word_indices = []
    word_topics = []

    for word_idx, topic_scores in word_to_topics.items():
        # Find topic with highest score for this word
        dominant_topic = max(topic_scores, key=lambda x: x[1])[0]
        word_indices.append(word_idx)
        word_topics.append(dominant_topic)

    console.print_debug(f"  Total unique words across all topics: {len(word_indices)}", tag="VISUALIZATION")

    return word_indices, word_topics


def _remove_outliers_percentile(tsne_data: pd.DataFrame, percentile: float = 0.95,
                                console: Optional[ConsoleManager] = None) -> pd.DataFrame:
    """
    Remove outlier points that are far from the main cluster using percentile-based filtering.

    Args:
        tsne_data: DataFrame with x, y, hue columns
        percentile: Fraction of points to keep (default 0.95 = keep 95% closest to center)
        console: Console manager for logging

    Returns:
        Filtered DataFrame with outliers removed
    """
    _console = console or get_console()

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
        _console.print_debug(f"Removed {n_removed} outlier points (keeping {percentile*100:.1f}% closest to center)", tag="VISUALIZATION")
        _console.print_debug(f"Distance threshold: {threshold:.2f}", tag="VISUALIZATION")

    return filtered_data


def _apply_density_based_reduction(tsne_data: pd.DataFrame, density_threshold: float = 0.1,
                                   console: Optional[ConsoleManager] = None) -> pd.DataFrame:
    """
    Reduce points only in areas where they are too densely clustered.

    Args:
        tsne_data: DataFrame with x, y, hue columns
        density_threshold: Minimum distance threshold for keeping points
        console: Console manager for logging

    Returns:
        DataFrame with density-reduced points
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans

    _console = console or get_console()

    if len(tsne_data) <= 500:  # Don't reduce small datasets
        return tsne_data

    try:
        _console.print_debug(f"Optimizing visualization with {len(tsne_data):,} points...", tag="VISUALIZATION")

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
            _console.print_debug(f"  Topic {int(hue)}: {len(topic_data):,} -> {len(reduced_topic):,} points ({reduction_ratio:.1%})", tag="VISUALIZATION")

        result = pd.concat(reduced_dfs, ignore_index=True)
        total_reduction = len(result) / len(tsne_data)
        _console.print_debug(f"Optimization complete: {len(tsne_data):,} -> {len(result):,} points ({total_reduction:.1%} retained)", tag="VISUALIZATION")

        return result

    except Exception as e:
        _console.print_warning(f"Optimization failed, using all points: {e}", tag="VISUALIZATION")
        return tsne_data


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

    Args:
        color1: RGB(A) tuple
        color2: RGB(A) tuple

    Returns:
        Float distance value
    """
    r1, g1, b1 = color1[:3]
    r2, g2, b2 = color2[:3]

    # Simple RGB distance
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
