"""
Visualizations for Topic Similarity Analysis.

Generates heatmaps, dendrograms, and network graphs for topic similarity matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Optional, Dict
import networkx as nx
from pathlib import Path

from ..console.console_manager import ConsoleManager, get_console


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    topic_names: List[str],
    output_path: str,
    title: str = "Topic Similarity Heatmap",
    figsize: tuple = (12, 10),
    cmap: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "Cosine Similarity"
) -> None:
    """
    Create a heatmap visualization of topic similarity matrix.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of shape (n_topics, n_topics) with similarity values.
    topic_names : List[str]
        Names for each topic (used for axis labels).
    output_path : str
        Path where the plot will be saved.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size (width, height) in inches.
    cmap : str
        Matplotlib colormap name.
    annot : bool
        If True, write similarity values in each cell.
    fmt : str
        Format string for annotations.
    vmin : float
        Minimum value for colormap.
    vmax : float
        Maximum value for colormap.
    cbar_label : str
        Label for the colorbar.
    """
    plt.figure(figsize=figsize)

    # Create heatmap
    ax = sns.heatmap(
        similarity_matrix,
        xticklabels=topic_names,
        yticklabels=topic_names,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': cbar_label}
    )

    # Customize
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Topics", fontsize=12, fontweight='bold')
    plt.ylabel("Topics", fontsize=12, fontweight='bold')

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    _console = get_console()
    _console.print_debug(f"Saved similarity heatmap to: {output_path}", tag="VISUALIZATION")


def plot_similarity_dendrogram(
    similarity_matrix: np.ndarray,
    topic_names: List[str],
    output_path: str,
    title: str = "Topic Similarity Dendrogram",
    figsize: tuple = (14, 8),
    method: str = 'average',
    color_threshold: Optional[float] = None
) -> None:
    """
    Create a dendrogram showing hierarchical clustering of topics.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of shape (n_topics, n_topics) with similarity values.
    topic_names : List[str]
        Names for each topic (used for labels).
    output_path : str
        Path where the plot will be saved.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size (width, height) in inches.
    method : str
        Linkage method for hierarchical clustering.
        Options: 'single', 'complete', 'average', 'ward', etc.
    color_threshold : Optional[float]
        Threshold for coloring clusters. If None, uses default.
    """
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix

    # Ensure it's a valid distance matrix (symmetric, zero diagonal)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed distance matrix for linkage
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)

    # Create dendrogram
    plt.figure(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=topic_names,
        color_threshold=color_threshold,
        above_threshold_color='gray',
        leaf_font_size=10
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Topics", fontsize=12, fontweight='bold')
    plt.ylabel("Distance (1 - Similarity)", fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    _console = get_console()
    _console.print_debug(f"Saved similarity dendrogram to: {output_path}", tag="VISUALIZATION")


def plot_topic_similarity_network(
    similarity_matrix: np.ndarray,
    topic_names: List[str],
    output_path: str,
    threshold: float = 0.5,
    title: str = "Topic Similarity Network",
    figsize: tuple = (14, 14),
    node_size: int = 3000,
    font_size: int = 10,
    edge_width_scale: float = 5.0,
    layout: str = 'spring'
) -> None:
    """
    Create a network graph showing topic relationships.

    Nodes represent topics, and edges represent similarities above a threshold.
    Edge thickness represents similarity strength.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of shape (n_topics, n_topics) with similarity values.
    topic_names : List[str]
        Names for each topic (used for node labels).
    output_path : str
        Path where the plot will be saved.
    threshold : float
        Minimum similarity to draw an edge. Range: [0, 1].
    title : str
        Title for the plot.
    figsize : tuple
        Figure size (width, height) in inches.
    node_size : int
        Size of nodes in the graph.
    font_size : int
        Font size for node labels.
    edge_width_scale : float
        Scale factor for edge widths.
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai', etc.
    """
    # Create graph
    G = nx.Graph()

    # Add nodes
    for i, name in enumerate(topic_names):
        G.add_node(i, label=name)

    # Add edges for similarities above threshold
    edges_added = 0
    for i in range(len(topic_names)):
        for j in range(i + 1, len(topic_names)):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                G.add_edge(i, j, weight=similarity)
                edges_added += 1

    _console = get_console()
    if edges_added == 0:
        _console.print_warning(f"No edges above threshold {threshold}. Lowering threshold to 0.3", tag="VISUALIZATION")
        threshold = 0.3
        for i in range(len(topic_names)):
            for j in range(i + 1, len(topic_names)):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    G.add_edge(i, j, weight=similarity)

    # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Create plot
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=node_size,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )

    # Draw edges with varying thickness based on similarity
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        width=[w * edge_width_scale for w in weights],
        alpha=0.6,
        edge_color=weights,
        edge_cmap=plt.cm.YlOrRd,
        edge_vmin=threshold,
        edge_vmax=1.0
    )

    # Draw labels
    labels = {i: name for i, name in enumerate(topic_names)}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=font_size,
        font_weight='bold'
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    # Add colorbar for edge weights
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd,
        norm=plt.Normalize(vmin=threshold, vmax=1.0)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Similarity', fontsize=12, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    _console.print_debug(f"Saved topic similarity network to: {output_path}", tag="VISUALIZATION")


def plot_similarity_distribution(
    similarity_matrix: np.ndarray,
    output_path: str,
    title: str = "Topic Similarity Distribution",
    figsize: tuple = (10, 6),
    bins: int = 30
) -> None:
    """
    Plot histogram of pairwise topic similarities.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of shape (n_topics, n_topics) with similarity values.
    output_path : str
        Path where the plot will be saved.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size (width, height) in inches.
    bins : int
        Number of histogram bins.
    """
    # Get off-diagonal values (exclude self-similarities)
    n = similarity_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    similarities = similarity_matrix[mask]

    # Create plot
    plt.figure(figsize=figsize)

    # Histogram
    plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    # Add vertical lines for statistics
    mean_sim = similarities.mean()
    median_sim = np.median(similarities)

    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.axvline(median_sim, color='green', linestyle='--', linewidth=2, label=f'Median: {median_sim:.3f}')

    # Labels and title
    plt.xlabel("Similarity Score", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    _console = get_console()
    _console.print_debug(f"Saved similarity distribution plot to: {output_path}", tag="VISUALIZATION")


def plot_combined_similarity_analysis(
    similarity_matrix: np.ndarray,
    topic_names: List[str],
    output_dir: str,
    dataset_name: str = "topics",
    threshold: float = 0.5,
    create_network: bool = True,
    create_dendrogram: bool = True,
    create_distribution: bool = True
) -> Dict[str, str]:
    """
    Create a comprehensive set of similarity visualizations.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of shape (n_topics, n_topics) with similarity values.
    topic_names : List[str]
        Names for each topic.
    output_dir : str
        Directory where plots will be saved.
    dataset_name : str
        Name of the dataset (used in filenames).
    threshold : float
        Similarity threshold for network graph.
    create_network : bool
        If True, create network graph.
    create_dendrogram : bool
        If True, create dendrogram.
    create_distribution : bool
        If True, create distribution histogram.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot type to file path.
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # 1. Similarity heatmap (always created)
    heatmap_path = output_dir / f"{dataset_name}_topic_similarity_heatmap.png"
    plot_similarity_heatmap(
        similarity_matrix,
        topic_names,
        str(heatmap_path),
        title=f"Topic Similarity Heatmap - {dataset_name}"
    )
    output_paths['heatmap'] = str(heatmap_path)

    # 2. Dendrogram (optional)
    if create_dendrogram:
        dendrogram_path = output_dir / f"{dataset_name}_topic_similarity_dendrogram.png"
        plot_similarity_dendrogram(
            similarity_matrix,
            topic_names,
            str(dendrogram_path),
            title=f"Topic Similarity Dendrogram - {dataset_name}"
        )
        output_paths['dendrogram'] = str(dendrogram_path)

    # 3. Network graph (optional)
    if create_network:
        network_path = output_dir / f"{dataset_name}_topic_similarity_network.png"
        plot_topic_similarity_network(
            similarity_matrix,
            topic_names,
            str(network_path),
            threshold=threshold,
            title=f"Topic Similarity Network - {dataset_name}"
        )
        output_paths['network'] = str(network_path)

    # 4. Distribution histogram (optional)
    if create_distribution:
        distribution_path = output_dir / f"{dataset_name}_topic_similarity_distribution.png"
        plot_similarity_distribution(
            similarity_matrix,
            str(distribution_path),
            title=f"Topic Similarity Distribution - {dataset_name}"
        )
        output_paths['distribution'] = str(distribution_path)

    return output_paths


def plot_redundancy_report(
    redundant_pairs: List[Dict],
    output_path: str,
    title: str = "Redundant Topic Pairs",
    figsize: tuple = (10, 8),
    max_pairs: int = 20
) -> None:
    """
    Create a bar chart showing redundant topic pairs.

    Parameters
    ----------
    redundant_pairs : List[Dict]
        List of redundant pairs from find_redundant_topics().
    output_path : str
        Path where the plot will be saved.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size (width, height) in inches.
    max_pairs : int
        Maximum number of pairs to display.
    """
    _console = get_console()
    if not redundant_pairs:
        _console.print_debug("No redundant pairs to plot.", tag="VISUALIZATION")
        return

    # Limit to top pairs
    pairs_to_plot = redundant_pairs[:max_pairs]

    # Create labels and values
    labels = [
        f"{p['topic_1_name']} ↔ {p['topic_2_name']}"
        for p in pairs_to_plot
    ]
    similarities = [p['similarity'] for p in pairs_to_plot]

    # Create plot
    plt.figure(figsize=figsize)

    # Horizontal bar chart
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(labels)))
    plt.barh(range(len(labels)), similarities, color=colors, edgecolor='black')

    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Similarity Score", fontsize=12, fontweight='bold')
    plt.ylabel("Topic Pairs", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(similarities):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    _console.print_debug(f"Saved redundancy report to: {output_path}", tag="VISUALIZATION")
