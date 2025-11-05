import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Union, List, Tuple
import math


def topic_space_graph_output(w: np.ndarray, h: np.ndarray,
                            s_matrix: Optional[np.ndarray] = None,
                            output_dir: Optional[Union[str, Path]] = None,
                            table_name: str = "topic_space_plot",
                            top_k: int = 3,
                            min_probability: float = 0.05,
                            positioning: str = "radial") -> Optional[str]:
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
    print(f"\n🎯 Starting Topic-Space Fuzzy Classification Visualization")
    print(f"📊 Input Data:")
    print(f"  • Documents: {w.shape[0] if w is not None else 'None'}")
    print(f"  • Topics: {h.shape[0] if h is not None else 'None'}")
    print(f"  • Top-K topics per doc: {top_k}")
    print(f"  • Positioning: {positioning}")
    print(f"  • Output: {table_name}")

    if w is None or h is None:
        print("⚠️  Error: Invalid input matrices for topic-space visualization")
        return None

    # Note: This visualization assumes standard NMF with direct document-topic probabilities
    # NMTF models may not visualize correctly in this fuzzy classification view
    if s_matrix is not None:
        print("⚠️  Warning: NMTF detected. This fuzzy classification view is designed for standard NMF.")
        print("   NMTF topics may not display accurately in this visualization.")

    if w.shape[0] < 1:
        print("⚠️  Error: Need at least 1 document for visualization")
        return None

    # Convert W to dense array if necessary
    if hasattr(w, 'toarray'):
        w_dense = w.toarray()
        print(f"🔄 Converted sparse matrix to dense: {w.shape} → {w_dense.shape}")
    else:
        w_dense = np.asarray(w)
        print(f"✅ Using matrix as-is: {w_dense.shape}")

    n_docs, n_topics = w_dense.shape

    # Calculate topic centers (always circular) - REDUCED RADIUS
    topic_centers = _calculate_topic_centers(n_topics)
    print(f"🎯 Placed {n_topics} topic centers in circular arrangement")

    # Position documents using specified method
    doc_positions, doc_data = _position_documents(w_dense, topic_centers, top_k, min_probability, positioning)
    print(f"📍 Positioned {len(doc_positions)} documents using {positioning} method")

    # Create the visualization
    return _create_topic_space_plot(doc_positions, doc_data, topic_centers,
                                  output_dir, table_name, n_topics, positioning)


def _calculate_topic_centers(n_topics: int) -> np.ndarray:
    """
    Calculate fixed positions for topic centers in circular arrangement.
    REDUCED RADIUS to bring topics closer to documents.

    Args:
        n_topics: Number of topics

    Returns:
        Array of shape (n_topics, 2) with topic center coordinates
    """
    # Further reduced radius to optimize space utilization with new center-filling approach
    radius = max(1.8, n_topics * 0.25)  # Smaller radius to work with new center-filling algorithm
    angles = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)

    centers = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])

    return centers


def _position_documents(w_dense: np.ndarray, topic_centers: np.ndarray,
                       top_k: int, min_probability: float, positioning: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Position documents using either radial or polar coordinate methods.

    Args:
        w_dense: Document-topic matrix
        topic_centers: Topic center coordinates
        top_k: Number of top topics to consider per document
        min_probability: Minimum probability threshold
        positioning: "radial" or "polar" positioning method

    Returns:
        Tuple of (document_positions, document_metadata)
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
    Position documents on radial lines extending from origin through topic centers.
    Distance from origin = topic probability strength.
    INCREASED SCALING to bring documents closer to topic centers.

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

    # Calculate the radius of the topic circle for distance normalization
    topic_radius = np.linalg.norm(topic_centers[0])  # Distance from origin to topic centers

    for doc_idx in range(n_docs):
        doc_probs = w_dense[doc_idx]

        # Get top-k topics for this document
        top_indices = np.argsort(doc_probs)[-top_k:][::-1]  # Descending order
        top_probs = doc_probs[top_indices]

        # Filter by minimum probability
        valid_mask = top_probs >= min_probability
        if not valid_mask.any():
            valid_mask = np.zeros_like(valid_mask)
            valid_mask[0] = True

        valid_indices = top_indices[valid_mask]
        valid_probs = top_probs[valid_mask]
        valid_probs = valid_probs / valid_probs.sum()  # Normalize

        # Calculate position using radial method
        if len(valid_indices) == 1:
            # Single topic: position on radial line from origin through topic center
            topic_idx = valid_indices[0]
            topic_angle = topic_angles[topic_idx]
            topic_strength = valid_probs[0]

            # NEW SCALING - allow documents from center to topic centers
            # Very strong topic membership = very close to center
            # Moderate topic membership = middle distance
            # Weak topic membership = closer to topic centers
            distance_from_origin = (0.05 + topic_strength * 0.9) * topic_radius  # Range: 5-95% of topic radius

            doc_pos = np.array([
                distance_from_origin * np.cos(topic_angle),
                distance_from_origin * np.sin(topic_angle)
            ])

        else:
            # Multiple topics: weighted position between topic directions
            # Calculate weighted angle using circular statistics
            x_component = 0
            y_component = 0

            for topic_idx, prob in zip(valid_indices, valid_probs):
                angle = topic_angles[topic_idx]
                x_component += prob * np.cos(angle)
                y_component += prob * np.sin(angle)

            # Resulting angle and strength
            weighted_angle = np.arctan2(y_component, x_component)
            resultant_strength = np.sqrt(x_component**2 + y_component**2)

            # NEW distance for multi-topic documents
            # High alignment (similar topics) = closer to topic centers
            # Low alignment (conflicting topics) = closer to center (mixed topic area)
            distance_from_origin = (0.1 + resultant_strength * 0.8) * topic_radius  # Range: 10-90% of topic radius

            doc_pos = np.array([
                distance_from_origin * np.cos(weighted_angle),
                distance_from_origin * np.sin(weighted_angle)
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
        # NEW APPROACH - fill center area based on topic certainty
        dominant_strength = valid_probs[0]  # Strength of dominant topic

        # Create gradient from center (high certainty) to edge (low certainty)
        if dominant_strength > 0.9:
            # Extremely strong single topic - very close to center (0-15% of topic radius)
            radius = topic_radius * (0.0 + 0.15 * (dominant_strength - 0.9) / 0.1)
        elif dominant_strength > 0.8:
            # Very strong single topic - close to center (15-25% of topic radius)
            radius = topic_radius * (0.15 + 0.10 * (dominant_strength - 0.8) / 0.1)
        elif dominant_strength > 0.6:
            # Strong topic - inner ring (25-45% of topic radius)
            radius = topic_radius * (0.25 + 0.20 * (dominant_strength - 0.6) / 0.2)
        elif dominant_strength > 0.4:
            # Moderate topic - middle ring (45-65% of topic radius)
            radius = topic_radius * (0.45 + 0.20 * (dominant_strength - 0.4) / 0.2)
        elif dominant_strength > 0.25:
            # Weak topic - outer ring (65-80% of topic radius)
            radius = topic_radius * (0.65 + 0.15 * (dominant_strength - 0.25) / 0.15)
        elif dominant_strength > 0.15:
            # Very weak topic - near topic centers (80-90% of topic radius)
            radius = topic_radius * (0.80 + 0.10 * (dominant_strength - 0.15) / 0.1)
        else:
            # Ultra-mixed topics - close to topic centers (90-95% of topic radius)
            radius = topic_radius * (0.90 + 0.05 * dominant_strength / 0.15)

        # Apply resultant strength to handle multi-topic documents
        # NEW approach - mixed topics go toward center, aligned topics stay out
        radius *= max(0.05, resultant_strength)  # Allow very close to center for mixed topics

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
        print(f"💾 Topic-space plot saved: {saved_path}")

    plt.show()

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
    # Add concentric circles - FURTHER REDUCED to match new center-filling approach
    max_radius = 2.5  # Further reduced to match new smaller topic center radius
    for radius in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]:
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
        print("📝 Note: scipy not available for topic territories")
    except Exception as e:
        print(f"📝 Note: Could not create topic territories: {e}")



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
    print(f"\n🎯 Topic-Space Visualization Summary:")
    print(f"📝 Total documents: {len(doc_data):,}")
    print(f"🎨 Number of topics: {n_topics}")

    # Topic distribution
    print(f"🔍 Document Distribution by Dominant Topic:")
    topic_counts = doc_data['dominant_topic'].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        percentage = (count / len(doc_data)) * 100
        print(f"  • Topic {topic_id + 1}: {count:,} documents ({percentage:.1f}%)")

    # Certainty statistics
    avg_certainty = doc_data['certainty'].mean()
    print(f"\n📊 Topic Assignment Certainty:")
    print(f"  • Average certainty: {avg_certainty:.3f}")
    print(f"  • High certainty docs (>0.8): {(doc_data['certainty'] > 0.8).sum():,}")
    print(f"  • Low certainty docs (<0.3): {(doc_data['certainty'] < 0.3).sum():,}")

    # Multi-topic documents
    multi_topic_docs = doc_data[doc_data['certainty'] < 0.5]
    if len(multi_topic_docs) > 0:
        print(f"  • Multi-topic documents: {len(multi_topic_docs):,} ({len(multi_topic_docs)/len(doc_data)*100:.1f}%)")