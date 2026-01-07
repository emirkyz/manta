"""
UMAP visualization for document-topic analysis in MANTA.

This module provides UMAP (Uniform Manifold Approximation and Projection) based
visualization of document-topic relationships, offering an alternative to t-SNE
with potentially better preservation of global structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Union, List
from ...utils.analysis import get_dominant_topics
from ..console.console_manager import ConsoleManager, get_console
from .visualization_helpers import _generate_distinct_colors


def umap_graph_output(w: np.ndarray, h: np.ndarray,
                      s_matrix: Optional[np.ndarray] = None,
                      output_dir: Optional[Union[str, Path]] = None,
                      table_name: str = "umap_plot",
                      n_neighbors: int = 15,
                      console: Optional[ConsoleManager] = None) -> Optional[str]:
    """
    Create UMAP visualizations for document-topic analysis.

    This function generates modern, aesthetically pleasing UMAP plots that show how documents
    cluster by topic in a 2D space. UMAP often preserves global structure better than t-SNE.

    Args:
        w: Document-topic matrix (W from NMF/LDA) - shape (n_docs, n_topics)
        h: Topic-word matrix (H from NMF/LDA) - shape (n_topics, n_words)
        s_matrix: Optional S matrix from NMTF (k×k). DEPRECATED: No longer used for reordering.
                 Kept for backwards compatibility.
        output_dir: Directory to save the plot (optional)
        table_name: Base name for the output file and plot title
        n_neighbors: Number of neighbors for UMAP (default 15). Controls local vs global structure.
                    Lower values (5-10): Focus on local structure, tighter clusters
                    Higher values (30-50): Focus on global structure, broader patterns
        console: Optional ConsoleManager for logging

    Returns:
        Path to saved plot file, or None if saving failed

    Features:
        - Modern, clean aesthetic with professional color schemes
        - Adaptive point sizing based on dataset size
        - Colored points by topic assignment with legend
        - High-resolution output suitable for publications
        - UMAP generally faster than t-SNE for large datasets
        - Better preservation of global structure

    Note:
        When used with NMTF, the S matrix is expected to be column-normalized (L1 norm)
        where each column sums to 1.0. This ensures consistent probability-like interpretation
        of topic relationships across visualizations.
    """
    # Input validation with console output
    _console = console or get_console()
    _console.print_debug("Starting UMAP Visualization", tag="VISUALIZATION")
    _console.print_debug(f"Input Data: Documents={w.shape[0] if w is not None else 'None'}, Topics={h.shape[0] if h is not None else 'None'}", tag="VISUALIZATION")
    _console.print_debug(f"Output: {table_name}", tag="VISUALIZATION")

    if w is None or h is None:
        _console.print_warning("Invalid input matrices for UMAP visualization", tag="VISUALIZATION")
        return None

    if w.shape[0] < 2:
        _console.print_warning("Need at least 2 documents for UMAP visualization", tag="VISUALIZATION")
        return None

    # Check for UMAP installation
    try:
        from umap import UMAP
    except ImportError:
        _console.print_warning(
            "UMAP not installed. Install with: pip install umap-learn",
            tag="VISUALIZATION"
        )
        return None

    _console.print_debug(f"Generating UMAP embedding for {w.shape[0]:,} documents and {h.shape[0]} topics...", tag="VISUALIZATION")

    # Convert W to dense array only if necessary
    if hasattr(w, 'toarray'):
        # It's a sparse matrix, convert to dense
        w_dense = w.toarray()
        _console.print_debug(f"Converted sparse matrix to dense: {w.shape} -> {w_dense.shape}", tag="VISUALIZATION")
    else:
        # Use np.asarray to avoid copying if already an array
        w_dense = np.asarray(w)
        _console.print_debug(f"Using matrix as-is: {w_dense.shape}", tag="VISUALIZATION")

    # Apply UMAP to document-topic matrix (W) with optimized parameters
    n_docs = w_dense.shape[0]

    # Validate n_neighbors
    if n_neighbors >= n_docs:
        # Adjust n_neighbors to be less than n_docs
        n_neighbors = max(2, min(15, n_docs - 1))
        _console.print_warning(
            f"n_neighbors adjusted to {n_neighbors} (must be < n_documents)",
            tag="VISUALIZATION"
        )

    _console.print_debug(
        f"UMAP parameters: n_neighbors={n_neighbors}, min_dist=0.1, metric=cosine",
        tag="VISUALIZATION"
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.5,
        metric='euclidean',
        random_state=42,
        n_components=2,
        spread=0.7
    )

    umap_embedding = umap_model.fit_transform(w_dense)
    umap_embedding = pd.DataFrame(umap_embedding, columns=['x', 'y'])

    # Use W directly - no reordering needed
    # Topic ordering is now sequential (Topic i = W column i)
    # This ensures consistency with word extraction across all visualizations
    w_for_topics = w_dense

    # Get dominant topics, filtering out zero-score documents
    dominant_topics = get_dominant_topics(w_for_topics, min_score=0.0)
    umap_embedding['hue'] = dominant_topics

    # Filter out documents with no dominant topic (marked as -1)
    valid_mask = umap_embedding['hue'] != -1
    excluded_count = (~valid_mask).sum()

    if excluded_count > 0:
        _console.print_debug(f"Excluded {excluded_count} documents with all zero topic scores from UMAP visualization", tag="VISUALIZATION")

    umap_embedding = umap_embedding[valid_mask].reset_index(drop=True)

    # Create the visualization with modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='white', edgecolor='none')

    data = umap_embedding

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

    # Point size multiplier (matching t-SNE style)
    point_size *= 1.5


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

    _console.print_debug(f"Visualization settings: {n_points} points, size={point_size:.1f}, alpha={alpha:.2f}", tag="VISUALIZATION")

    # Use maximally distinct colors like digital city maps
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
    title_text = f'Topic Distribution Visualization (UMAP)\n{table_name.replace("_", " ").title()}'
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25,
                 color='#2E3440', family='sans-serif')
    ax.set_xlabel('UMAP Component 1', fontsize=13, color='#4C566A', fontweight='medium')
    ax.set_ylabel('UMAP Component 2', fontsize=13, color='#4C566A', fontweight='medium')

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

        filename = f"{table_name}_umap_visualization.png"
        file_path = output_path / filename

        # High-quality save settings for professional output
        plt.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    pad_inches=0.2, format='png',
                    metadata={'Title': f'UMAP Visualization: {table_name}',
                              'Description': 'Generated by MANTA Topic Modeling'})
        saved_path = str(file_path)
        _console.print_debug(f"High-quality plot saved: {saved_path}", tag="VISUALIZATION")

    plt.show()

    # Print summary statistics
    _console.print_debug(f"UMAP Visualization Summary: {len(data):,} documents, {len(unique_topics)} topics", tag="VISUALIZATION")
    for topic_id in unique_topics:
        topic_count = len(data[data['hue'] == topic_id])
        percentage = (topic_count / len(data)) * 100
        _console.print_debug(f"  Topic {topic_id + 1}: {topic_count:,} documents ({percentage:.1f}%)", tag="VISUALIZATION")

    return saved_path
