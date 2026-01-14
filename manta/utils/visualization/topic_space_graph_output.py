import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Union, List, Tuple
import math
from ...utils.analysis import get_dominant_topics
from ..console.console_manager import ConsoleManager, get_console


def topic_space_graph_output(w: np.ndarray, h: np.ndarray,
                            output_dir: Optional[Union[str, Path]] = None,
                            table_name: str = "topic_space_plot",
                            top_k: int = 3,
                            min_probability: float = 0.05,
                            positioning: str = "radial",
                            console: Optional[ConsoleManager] = None) -> Optional[str]:
    """
    Create topic-space visualization where documents are positioned based on
    weighted combinations of their topic memberships.

    This approach places topic centers in fixed positions and positions each document
    as a weighted average of its top-k topics' centers, creating intuitive fuzzy
    classification visualization.

    Args:
        w: Document-topic matrix (W from NMF/LDA) - shape (n_docs, n_topics)
        h: Topic-word matrix (H from NMF/LDA) - shape (n_topics, n_words)
        output_dir: Directory to save the plot (optional)
        table_name: Base name for the output file and plot title
        top_k: Number of top topics to consider per document (default: 3)
        min_probability: Minimum topic probability to include (default: 0.05)
        positioning: Document positioning method ("radial" or "polar")

    Returns:
        Path to saved plot file, or None if saving failed

    Features:
        - Fixed topic centers with clear labels
        - Document positioning based on weighted topic probabilities
        - Fuzzy classification showing multi-topic memberships
        - Point size based on topic certainty (inverse entropy)
        - Clear topic territories and boundaries
    """
    _console = console or get_console()
    _console.print_debug("Starting Topic-Space Fuzzy Classification Visualization", tag="VISUALIZATION")
    _console.print_debug(f"Input Data: Documents={w.shape[0] if w is not None else 'None'}, Topics={h.shape[0] if h is not None else 'None'}", tag="VISUALIZATION")
    _console.print_debug(f"Top-K={top_k}, Positioning={positioning}, Output={table_name}", tag="VISUALIZATION")

    if w is None or h is None:
        _console.print_warning("Invalid input matrices for topic-space visualization", tag="VISUALIZATION")
        return None

    if w.shape[0] < 1:
        _console.print_warning("Need at least 1 document for visualization", tag="VISUALIZATION")
        return None

    # Convert W to dense array if necessary
    if hasattr(w, 'toarray'):
        w_dense = w.toarray()
        _console.print_debug(f"Converted sparse matrix to dense: {w.shape} -> {w_dense.shape}", tag="VISUALIZATION")
    else:
        w_dense = np.asarray(w)
        _console.print_debug(f"Using matrix as-is: {w_dense.shape}", tag="VISUALIZATION")

    n_docs, n_topics = w_dense.shape

    # Filter out documents with all zero topic scores
    dominant_topics = get_dominant_topics(w_dense, min_score=0.0)
    valid_mask = dominant_topics != -1
    excluded_count = np.sum(~valid_mask)

    if excluded_count > 0:
        _console.print_debug(f"Excluded {excluded_count} documents with all zero topic scores from visualization", tag="VISUALIZATION")

    # Filter to only valid documents
    w_filtered = w_dense[valid_mask]

    if w_filtered.shape[0] == 0:
        _console.print_warning("No valid documents to visualize after filtering", tag="VISUALIZATION")
        return None

    # Calculate topic centers (always circular) - REDUCED RADIUS
    topic_centers = _calculate_topic_centers(n_topics)
    _console.print_debug(f"Placed {n_topics} topic centers in circular arrangement", tag="VISUALIZATION")

    # Position documents using specified method
    doc_positions, doc_data = _position_documents(w_filtered, topic_centers, top_k, min_probability, positioning)
    _console.print_debug(f"Positioned {len(doc_positions)} documents using {positioning} method", tag="VISUALIZATION")

    # Create the visualization
    return _create_topic_space_plot(doc_positions, doc_data, topic_centers,
                                  output_dir, table_name, n_topics, positioning)


def _calculate_topic_centers(n_topics: int) -> np.ndarray:
    """
    Calculate fixed positions for topic centers in circular arrangement.
    REDUCED RADIUS to bring topics closer to documents.
    Topic 1 starts at left (π radians) and goes clockwise.

    Args:
        n_topics: Number of topics

    Returns:
        Array of shape (n_topics, 2) with topic center coordinates
    """
    # MUCH smaller radius - reduced from max(4.0, n_topics * 0.6) to bring topics closer
    radius = max(2.5, n_topics * 0.3)  # Reduced radius significantly

    # Start at π (left side) and go clockwise (negative direction)
    start_angle = np.pi  # Left side
    angles = start_angle - np.linspace(0, 2 * np.pi, n_topics, endpoint=False)

    centers = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])

    return centers

def _position_documents(w_dense: np.ndarray, topic_centers: np.ndarray,
                       top_k: int, min_probability: float, positioning: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Position documents using either radial or polar coordinate methods.
    (This function remains the same, just calls the modified functions below)
    """
    if positioning == "radial":
        return _position_documents_radial(w_dense, topic_centers, top_k, min_probability)
    elif positioning == "polar":
        return _position_documents_polar(w_dense, topic_centers, top_k, min_probability)
    else:
        raise ValueError(f"Unknown positioning method: {positioning}")


def _position_documents_radial(w_dense: np.ndarray, topic_centers: np.ndarray,
                              top_k: int, min_probability: float) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    MODIFIED: Position documents on radial lines with improved distribution.
    This version eliminates the "dead space" in the center by scaling distance
    from the origin based on topic certainty. It also adds jitter to prevent
    overplotting.
    """
    n_docs, n_topics = w_dense.shape
    doc_positions = []
    doc_metadata = []

    topic_angles = np.arctan2(topic_centers[:, 1], topic_centers[:, 0])
    topic_radius = np.linalg.norm(topic_centers[0])

    for doc_idx in range(n_docs):
        doc_probs = w_dense[doc_idx]
        top_indices = np.argsort(doc_probs)[-top_k:][::-1]
        top_probs = doc_probs[top_indices]

        valid_mask = top_probs >= min_probability
        if not valid_mask.any():
            valid_mask = np.zeros_like(valid_mask); valid_mask[0] = True

        valid_indices = top_indices[valid_mask]
        valid_probs = top_probs[valid_mask]
        valid_probs /= valid_probs.sum()

        # Calculate weighted angle and resultant strength (certainty)
        x_comp = np.sum(valid_probs * np.cos(topic_angles[valid_indices]))
        y_comp = np.sum(valid_probs * np.sin(topic_angles[valid_indices]))

        weighted_angle = np.arctan2(y_comp, x_comp)
        resultant_strength = np.sqrt(x_comp**2 + y_comp**2)

        # --- KEY CHANGE START ---
        # Map resultant strength (certainty) to distance from origin.
        # A strength of 0 (highly mixed) is at the center.
        # A strength of 1 (pure topic) is near the topic center.
        # The exponent (0.5) spreads the points out more evenly.
        scaling_exponent = 0.5
        distance_from_origin = (resultant_strength ** scaling_exponent) * topic_radius

        # Add a small amount of random jitter to avoid overplotting
        jitter_strength = 0.025 * topic_radius # 2.5% of the radius
        jitter_x = np.random.randn() * jitter_strength
        jitter_y = np.random.randn() * jitter_strength

        doc_pos = np.array([
            distance_from_origin * np.cos(weighted_angle) + jitter_x,
            distance_from_origin * np.sin(weighted_angle) + jitter_y
        ])
        # --- KEY CHANGE END ---

        doc_positions.append(doc_pos)
        certainty = _calculate_certainty(doc_probs)
        doc_metadata.append({
            'doc_id': doc_idx, 'x': doc_pos[0], 'y': doc_pos[1],
            'dominant_topic': top_indices[0], 'dominant_prob': top_probs[0],
            'certainty': certainty, 'top_topics': valid_indices.tolist(),
            'top_probs': valid_probs.tolist()
        })

    return np.array(doc_positions), pd.DataFrame(doc_metadata)


import numpy as np
import pandas as pd
from typing import Tuple

def _position_documents_polar(w_dense: np.ndarray, topic_centers: np.ndarray,
                             top_k: int, min_probability: float) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Position documents using polar coordinate mapping with concentric rings.
    Angle = weighted average of topic angles, Radius = topic strength.
    INCREASED MINIMUM RADIUS to bring documents closer to topic centers.

    Args:
        w_dense: Document-topic matrix
        topic_centers: Topic center coordinates
        top_k: Number of top topics to consider per document
        min_probability: Minimum probability threshold

    Returns:
        Tuple of (document_positions, document_metadata)
    """
    n_docs, n_topics = w_dense.shape
    doc_positions = []
    doc_metadata = []

    # Calculate topic angles from their positions
    topic_angles = []
    for center in topic_centers:
        angle = np.arctan2(center[1], center[0])
        topic_angles.append(angle)
    topic_angles = np.array(topic_angles)

    # Calculate the radius of the topic circle for normalization
    topic_radius = np.linalg.norm(topic_centers[0])

    for doc_idx in range(n_docs):
        doc_probs = w_dense[doc_idx]

        # Get top-k topics for this document
        top_indices = np.argsort(doc_probs)[-top_k:][::-1]
        top_probs = doc_probs[top_indices]

        # Filter by minimum probability
        valid_mask = top_probs >= min_probability
        if not valid_mask.any():
            valid_mask = np.zeros_like(valid_mask)
            valid_mask[0] = True

        valid_indices = top_indices[valid_mask]
        valid_probs = top_probs[valid_mask]
        valid_probs = valid_probs / valid_probs.sum()

        # Calculate weighted angle using proper circular statistics
        x_component = 0
        y_component = 0

        for topic_idx, prob in zip(valid_indices, valid_probs):
            angle = topic_angles[topic_idx]
            x_component += prob * np.cos(angle)
            y_component += prob * np.sin(angle)

        weighted_angle = np.arctan2(y_component, x_component)
        resultant_strength = np.sqrt(x_component**2 + y_component**2)

        # Calculate radius for concentric ring positioning
        # SIGNIFICANTLY INCREASED minimum distances to bring documents closer to topic centers
        dominant_strength = valid_probs[0]  # Strength of dominant topic

        # Create concentric rings with MUCH higher minimum radius
        if dominant_strength > 0.8:
            # Very strong single topic - very close to topic centers (85-95% of topic radius)
            radius = topic_radius * (0.85 + 0.10 * (dominant_strength - 0.8) / 0.2)
        elif dominant_strength > 0.6:
            # Strong topic - close to topic centers (75-85% of topic radius)
            radius = topic_radius * (0.75 + 0.10 * (dominant_strength - 0.6) / 0.2)
        elif dominant_strength > 0.4:
            # Moderate topic - moderately close (65-75% of topic radius)
            radius = topic_radius * (0.65 + 0.10 * (dominant_strength - 0.4) / 0.2)
        elif dominant_strength > 0.25:
            # Weak topic - still reasonably close (55-65% of topic radius)
            radius = topic_radius * (0.55 + 0.10 * (dominant_strength - 0.25) / 0.15)
        elif dominant_strength > 0.15:
            # Very weak topic - inner area but not too close to center (45-55% of topic radius)
            radius = topic_radius * (0.45 + 0.10 * (dominant_strength - 0.15) / 0.1)
        elif dominant_strength > 0.05:
            # Mixed topics - inner area (35-45% of topic radius)
            radius = topic_radius * (0.35 + 0.10 * (dominant_strength - 0.05) / 0.1)
        else:
            # Ultra-mixed topics - still substantial distance from center (25-35% of topic radius)
            radius = topic_radius * (0.25 + 0.10 * dominant_strength / 0.05)

        # Apply resultant strength to handle multi-topic documents
        # INCREASED minimum constraint to keep documents away from center
        radius *= max(0.3, resultant_strength)  # Increased from 0.05 to 0.3

        # Convert to Cartesian coordinates
        doc_pos = np.array([
            radius * np.cos(weighted_angle),
            radius * np.sin(weighted_angle)
        ])

        doc_positions.append(doc_pos)

        # Calculate topic certainty
        certainty = _calculate_certainty(doc_probs)

        doc_metadata.append({
            'doc_id': doc_idx,
            'x': doc_pos[0],
            'y': doc_pos[1],
            'dominant_topic': top_indices[0],
            'dominant_prob': top_probs[0],
            'certainty': certainty,
            'top_topics': valid_indices.tolist(),
            'top_probs': valid_probs.tolist()
        })

    return np.array(doc_positions), pd.DataFrame(doc_metadata)


def _calculate_certainty(doc_probs: np.ndarray) -> float:
    """
    Calculate topic certainty (inverse entropy).

    Args:
        doc_probs: Topic probability distribution for a document

    Returns:
        Certainty value between 0 and 1
    """
    nonzero_probs = doc_probs[doc_probs > 1e-10]
    if len(nonzero_probs) > 1:
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
        certainty = 1.0 / (1.0 + entropy)  # Scale to [0, 1]
    else:
        certainty = 1.0  # Perfect certainty

    return certainty


def _create_topic_space_plot(doc_positions: np.ndarray, doc_data: pd.DataFrame,
                           topic_centers: np.ndarray, output_dir: Optional[Union[str, Path]],
                           table_name: str, n_topics: int, positioning: str) -> Optional[str]:
    """
    Create the topic-space visualization plot.

    Args:
        doc_positions: Document position coordinates
        doc_data: Document metadata DataFrame
        topic_centers: Topic center coordinates
        output_dir: Output directory
        table_name: Plot title base name
        n_topics: Number of topics

    Returns:
        Path to saved plot or None
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), facecolor='white')

    # Generate colors optimized for circular topic arrangement
    distinct_colors = _get_optimized_topic_colors(n_topics)

    # Plot documents
    for _, doc in doc_data.iterrows():
        topic_id = doc['dominant_topic']
        color = distinct_colors[topic_id] if topic_id < len(distinct_colors) else (0.5, 0.5, 0.5, 1.0)

        # Fixed uniform point size for cleaner visualization
        point_size = 25

        # Alpha based on dominant probability (clamp to valid range)
        alpha = min(1.0, 0.3 + doc['dominant_prob'] * 0.7)  # Range: 0.3-1.0

        ax.scatter(doc['x'], doc['y'], s=point_size, c=[color],
                  alpha=alpha, edgecolors='white', linewidths=0.5)

    # Plot topic centers
    for topic_idx, center in enumerate(topic_centers):
        color = distinct_colors[topic_idx] if topic_idx < len(distinct_colors) else (0.5, 0.5, 0.5, 1.0)

        # Large marker for topic center
        ax.scatter(center[0], center[1], s=400, c=[color],
                  marker='s', edgecolors='black', linewidths=2,
                  alpha=0.9, zorder=10)

        # Topic label
        ax.annotate(f'T{topic_idx + 1}', (center[0], center[1]),
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   color='white', zorder=11)

    # Add Voronoi-like regions (optional)
    _add_topic_territories(ax, topic_centers, distinct_colors)

    # Add circular grid lines for better depth perception - ADJUSTED FOR SMALLER RADIUS
    _add_circular_grid(ax, positioning)

    # Styling
    title_method = positioning.capitalize()
    ax.set_title(f'Topic-Space Fuzzy Classification ({title_method})\n{table_name.replace("_", " ").title()}',
                fontsize=20, fontweight='bold', pad=30, color='#2E3440')

    ax.set_xlabel('Topic Space X', fontsize=14, color='#4C566A', fontweight='medium')
    ax.set_ylabel('Topic Space Y', fontsize=14, color='#4C566A', fontweight='medium')

    # Clean styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E9F0')
    ax.spines['bottom'].set_color('#E5E9F0')

    # Equal aspect ratio for undistorted topic relationships
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save the plot
    saved_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{table_name}_topic_space_visualization.png"
        file_path = output_path / filename

        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   pad_inches=0.2, format='png',
                   metadata={'Title': f'Topic-Space Visualization: {table_name}',
                            'Description': 'Generated by MANTA Topic Modeling'})
        saved_path = str(file_path)
        _console = get_console()
        _console.print_debug(f"Topic-space plot saved: {saved_path}", tag="VISUALIZATION")

    #plt.show()

    # Print summary statistics
    _print_summary_statistics(doc_data, n_topics)

    return saved_path


def _add_circular_grid(ax, positioning: str):
    """
    Add circular grid lines to enhance the circular visualization.
    ADJUSTED for smaller topic radius.

    Args:
        ax: Matplotlib axis
        positioning: Positioning method for appropriate grid styling
    """
    # Add concentric circles - REDUCED maximum radius to match new topic positioning
    max_radius = 3.0  # Reduced from 5.0 to match new smaller topic center radius
    for radius in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if radius <= max_radius:
            circle = plt.Circle((0, 0), radius, fill=False,
                              color='lightgray', alpha=0.3, linewidth=0.5)
            ax.add_patch(circle)

    # Add radial lines for radial positioning
    if positioning == "radial":
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x_end = max_radius * np.cos(angle)
            y_end = max_radius * np.sin(angle)
            ax.plot([0, x_end], [0, y_end], 'lightgray', alpha=0.2, linewidth=0.5)


def _add_topic_territories(ax, topic_centers: np.ndarray, colors: List[tuple]):
    """
    Add subtle background regions showing topic territories using Voronoi-like approach.

    Args:
        ax: Matplotlib axis
        topic_centers: Topic center coordinates
        colors: Topic colors
    """
    try:
        from scipy.spatial import Voronoi
        import matplotlib.patches as patches

        # Create Voronoi diagram
        vor = Voronoi(topic_centers)

        # Plot finite regions with subtle transparency
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            if all(v >= 0 for v in simplex):
                polygon = vor.vertices[simplex]

                # Determine which topic this region belongs to
                topic_idx = pointidx[0] if len(pointidx) > 0 else 0
                if topic_idx < len(colors):
                    color = colors[topic_idx]

                    # Add subtle background region
                    poly = patches.Polygon(polygon, alpha=0.1, facecolor=color,
                                         edgecolor=color, linewidth=0.5)
                    ax.add_patch(poly)

    except ImportError:
        _console = get_console()
        _console.print_debug("scipy not available for topic territories", tag="VISUALIZATION")
    except Exception as e:
        _console = get_console()
        _console.print_debug(f"Could not create topic territories: {e}", tag="VISUALIZATION")



def _get_optimized_topic_colors(n_topics: int) -> List[tuple]:
    """
    Generate colors optimized for circular topic arrangement to maximize contrast between neighbors.
    Uses seaborn's perceptually uniform color palettes with circular optimization.

    Args:
        n_topics: Number of colors needed

    Returns:
        List of RGB tuples optimized for circular arrangement
    """
    try:
        import seaborn as sns
    except ImportError:
        # Fallback to matplotlib if seaborn not available
        return _generate_distinct_colors(n_topics)

    if n_topics <= 1:
        return [(0.12, 0.47, 0.71, 1.0)]  # Single default blue

    # Choose optimal palette based on topic count
    if n_topics <= 10:
        # Use "husl" for small counts - provides maximally distinct hues
        base_colors = sns.color_palette("husl", n_topics)
    elif n_topics <= 20:
        # Use "tab20" for medium counts - more colors available
        base_colors = sns.color_palette("tab20", n_topics)
    else:
        # For large counts, use "husl" with optimal spacing
        base_colors = sns.color_palette("husl", n_topics)

    # Convert to RGBA tuples
    rgba_colors = []
    for color in base_colors:
        if len(color) == 3:  # RGB
            rgba_colors.append(color + (1.0,))  # Add alpha
        else:  # Already RGBA
            rgba_colors.append(color)

    # For circular arrangement, reorder colors to maximize neighbor contrast
    if n_topics >= 3:
        reordered_colors = _reorder_for_circular_contrast(rgba_colors, n_topics)
        return reordered_colors

    return rgba_colors


def _reorder_for_circular_contrast(colors: List[tuple], n_topics: int) -> List[tuple]:
    """
    Reorder colors to maximize contrast between circular neighbors.

    Args:
        colors: List of RGBA color tuples
        n_topics: Number of topics (for validation)

    Returns:
        Reordered list of colors optimized for circular arrangement
    """
    if n_topics <= 2:
        return colors

    # For odd numbers, use alternating pattern starting from 0
    # For even numbers, use alternating pattern with offset
    reordered = [None] * n_topics
    available_colors = list(colors)

    # Strategy: Fill positions by jumping around the circle
    # This ensures adjacent topics get maximally different colors

    # Start with position 0
    positions_to_fill = list(range(n_topics))

    # Create optimal ordering by spacing out color assignments
    step_size = max(2, n_topics // 3)  # Jump by at least 2, more for larger circles

    filled_positions = []
    color_idx = 0

    # First pass: fill every step_size positions
    for start_offset in range(step_size):
        pos = start_offset
        while pos < n_topics and color_idx < len(available_colors):
            if pos not in filled_positions:
                reordered[pos] = available_colors[color_idx]
                filled_positions.append(pos)
                color_idx += 1
            pos += step_size

    # Second pass: fill remaining positions
    for pos in range(n_topics):
        if pos not in filled_positions and color_idx < len(available_colors):
            reordered[pos] = available_colors[color_idx]
            color_idx += 1

    # Ensure no None values (fallback)
    for i, color in enumerate(reordered):
        if color is None:
            reordered[i] = available_colors[i % len(available_colors)]

    return reordered


def _generate_distinct_colors(n_topics: int) -> List[tuple]:
    """
    Generate maximally distinct colors for topics.

    Args:
        n_topics: Number of colors needed

    Returns:
        List of RGB tuples
    """
    import matplotlib.colors as mcolors

    # Predefined high-contrast palettes for common topic counts
    distinct_palettes = {
        2: ['#E31A1C', '#1F78B4'],
        3: ['#E31A1C', '#33A02C', '#1F78B4'],
        4: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4'],
        5: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A'],
        6: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99'],
        7: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928'],
        8: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F'],
    }

    if n_topics <= 8 and n_topics in distinct_palettes:
        colors = distinct_palettes[n_topics]
        return [mcolors.hex2color(color) for color in colors]

    # For larger numbers, use tab20 with optimized ordering
    if n_topics <= 20:
        import matplotlib.pyplot as plt
        base_colors = plt.cm.tab20(np.arange(20))
        optimized_order = [0, 10, 2, 12, 4, 14, 6, 16, 8, 18, 1, 11, 3, 13, 5, 15, 7, 17, 9, 19]
        reordered_colors = [base_colors[i] for i in optimized_order[:n_topics]]
        return [(r, g, b, a) for r, g, b, a in reordered_colors]
    import matplotlib.pyplot as plt
    # Fallback for very large numbers
    colors = plt.cm.Set3(np.linspace(0, 1, n_topics))
    return [(r, g, b, a) for r, g, b, a in colors]


def _print_summary_statistics(doc_data: pd.DataFrame, n_topics: int):
    """
    Print summary statistics for the topic-space visualization.

    Args:
        doc_data: Document metadata DataFrame
        n_topics: Number of topics
    """
    _console = get_console()
    _console.print_debug(f"Topic-Space Visualization Summary: {len(doc_data):,} documents, {n_topics} topics", tag="VISUALIZATION")

    # Topic distribution
    topic_counts = doc_data['dominant_topic'].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        percentage = (count / len(doc_data)) * 100
        _console.print_debug(f"  Topic {topic_id + 1}: {count:,} documents ({percentage:.1f}%)", tag="VISUALIZATION")

    # Certainty statistics
    avg_certainty = doc_data['certainty'].mean()
    _console.print_debug(f"Topic Assignment Certainty: avg={avg_certainty:.3f}, high(>0.8)={( doc_data['certainty'] > 0.8).sum():,}, low(<0.3)={(doc_data['certainty'] < 0.3).sum():,}", tag="VISUALIZATION")

    # Multi-topic documents
    multi_topic_docs = doc_data[doc_data['certainty'] < 0.5]
    if len(multi_topic_docs) > 0:
        _console.print_debug(f"Multi-topic documents: {len(multi_topic_docs):,} ({len(multi_topic_docs)/len(doc_data)*100:.1f}%)", tag="VISUALIZATION")