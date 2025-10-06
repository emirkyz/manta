"""
S Matrix Graph Relationship Visualization for NMTF.

This module visualizes the S matrix from Non-negative Matrix Tri-Factorization (NMTF)
as a network graph showing topic-to-topic relationships.

In NMTF: V ≈ W @ S @ H
The S matrix (k×k) represents relationships between latent factors (topics).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Union, Tuple, List
import json


def visualize_s_matrix_graph(
    s_matrix: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None,
    table_name: str = "s_matrix_graph",
    threshold: float = 0.01,
    layout: str = "circular",
    create_interactive: bool = True,
    create_heatmap: bool = True,
    figsize: Tuple[int, int] = (14, 14)
) -> dict:
    """
    Create graph visualizations of the S matrix showing topic relationships.

    This function visualizes the S matrix where values are normalized to [0, 1] range.
    Edge labels display the actual S matrix values, while hover information and statistics
    also show percentages relative to the total sum for additional context.

    Args:
        s_matrix: The S matrix from NMTF decomposition (k×k matrix, values in [0, 1] range)
        output_dir: Directory to save visualizations
        table_name: Base name for output files
        threshold: Minimum connection strength to display (filters weak edges, default 0.01)
        layout: Graph layout algorithm ("circular", "spring", "kamada_kawai", "hierarchical", "tree")
        create_interactive: Whether to create interactive plotly graph
        create_heatmap: Whether to create heatmap visualization
        figsize: Figure size for matplotlib plots

    Returns:
        Dictionary with paths to generated visualizations

    Visualizations created:
        - Network graph (specified layout): Shows edges with S value labels (0-1 range)
        - Hierarchical layout: Alternative layout for comparison
        - Custom matplotlib graph: Labels rotated along edges, skips short edges
        - Interactive graph: Hover shows strength and percentage (optional)
        - Heatmap: Cell annotations show percentage and absolute value (optional)

    Example:
        >>> s_matrix = nmf_output["S"]
        >>> paths = visualize_s_matrix_graph(
        ...     s_matrix=s_matrix,
        ...     output_dir="/output/path",
        ...     table_name="my_analysis",
        ...     threshold=0.1
        ... )
    """
    print(f"\n🔗 Creating S Matrix Graph Relationship Visualizations")
    print(f"📊 Matrix shape: {s_matrix.shape}")
    print(f"🔍 Threshold: {threshold}")
    print(f"📐 Layout: {layout}")

    # Convert to numpy array if needed
    if hasattr(s_matrix, 'toarray'):
        s_matrix = s_matrix.toarray()
    elif not isinstance(s_matrix, np.ndarray):
        s_matrix = np.array(s_matrix)

    n_topics = s_matrix.shape[0]

    # Setup output directory
    output_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    result_paths = {}

    # Create network graph visualization with specified layout
    network_path = _create_network_graph(
        s_matrix, output_path, table_name, threshold, layout, figsize
    )
    if network_path:
        result_paths[f'network_graph_{layout}'] = network_path

    # Create hierarchical layout version for comparison
    if layout != "hierarchical":
        print(f"\n📊 Creating hierarchical layout for comparison...")
        hierarchical_path = _create_network_graph(
            s_matrix, output_path, f"{table_name}_hierarchical", threshold, "hierarchical", figsize
        )
        if hierarchical_path:
            result_paths['network_graph_hierarchical'] = hierarchical_path

    # Create custom matplotlib version with smart label placement
    print(f"\n🎨 Creating custom matplotlib version...")
    custom_path = _create_custom_matplotlib_graph(
        s_matrix, output_path, table_name, threshold, layout, figsize
    )
    if custom_path:
        result_paths['custom_graph'] = custom_path

    # Create interactive visualization
    if create_interactive:
        interactive_path = _create_interactive_graph(
            s_matrix, output_path, table_name, threshold
        )
        if interactive_path:
            result_paths['interactive_graph'] = interactive_path

    # Create heatmap visualization
    if create_heatmap:
        heatmap_path = _create_s_matrix_heatmap(
            s_matrix, output_path, table_name, figsize
        )
        if heatmap_path:
            result_paths['heatmap'] = heatmap_path

    # Print summary statistics
    _print_s_matrix_statistics(s_matrix, threshold)

    return result_paths


def _create_network_graph(
    s_matrix: np.ndarray,
    output_path: Optional[Path],
    table_name: str,
    threshold: float,
    layout: str,
    figsize: Tuple[int, int]
) -> Optional[str]:
    """Create static network graph using matplotlib and networkx."""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️  NetworkX not installed. Skipping network graph. Install with: pip install networkx")
        return None

    n_topics = s_matrix.shape[0]

    # Calculate total sum of S matrix for percentage calculation
    total_sum = np.sum(s_matrix)
    print(f"📊 Total S matrix sum: {total_sum:.2f}")

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes (topics)
    for i in range(n_topics):
        G.add_node(i, label=f"T{i+1}")

    # Add edges (connections between topics)
    edges_added = 0
    edge_percentages = {}
    for col in range(n_topics):  # Column = source topic
        for row in range(n_topics):  # Row = target topic
            weight = s_matrix[row, col]
            if weight > threshold and row != col:  # Skip self-connections
                percentage = (weight / total_sum) * 100 if total_sum > 0 else 0
                G.add_edge(col, row, weight=weight, percentage=percentage)
                edge_percentages[(col, row)] = percentage
                edges_added += 1

    print(f"📈 Network: {n_topics} nodes, {edges_added} edges (threshold: {threshold})")

    if edges_added == 0:
        print(f"⚠️  No edges above threshold {threshold}. Try lowering the threshold.")
        return None

    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Choose layout
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "hierarchical":
        # Try graphviz hierarchical layout (dot)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            print("⚠️  Graphviz not available, using spring layout")
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "tree":
        # Try graphviz tree layout (twopi - radial tree)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
        except:
            print("⚠️  Graphviz not available, using kamada_kawai layout")
            pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)  # Fallback

    # Get edge weights for visualization
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 0

    # Normalize weights for visualization
    normalized_weights = [(w - min_weight) / (max_weight - min_weight + 1e-10) for w in weights]
    edge_widths = [1 + 4 * w for w in normalized_weights]  # Width: 1-5
    edge_alphas = [0.3 + 0.7 * w for w in normalized_weights]  # Alpha: 0.3-1.0

    # Draw nodes first with lower z-order to establish positions
    node_size = 1000
    node_colors = ['#88C0D0' for _ in range(n_topics)]

    # Draw edges with varying thickness and transparency (straight edges for better label positioning)
    # Set node_size parameter so arrows stop at node edges, not centers
    for (u, v), width, alpha, weight in zip(edges, edge_widths, edge_alphas, weights):
        nx.draw_networkx_edges(
            G, pos, [(u, v)],
            width=width,
            alpha=alpha,
            edge_color='#5E81AC',
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0',  # Straight edges for clean label positioning
            node_size=node_size,  # Arrows stop at node edge
            ax=ax
        )

    # Draw nodes on top with higher z-order
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.95,
        edgecolors='#2E3440',
        linewidths=2.5,
        ax=ax
    )

    # Draw labels
    labels = {i: f"T{i+1}" for i in range(n_topics)}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=12,
        font_weight='bold',
        font_color='white',
        ax=ax
    )

    # Draw edge labels with S matrix values (0-1 range)
    for (u, v) in G.edges():
        weight = G[u][v]['weight']
        percentage = G[u][v]['percentage']

        # Get positions of source and target nodes
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Calculate perpendicular offset to position label above the line
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            # Perpendicular vector (rotated 90 degrees)
            perp_x = -dy / length
            perp_y = dx / length

            # Offset distance (adjust this value to move labels further/closer to line)
            offset = 0.15

            # Label position above the line
            label_x = mid_x + perp_x * offset
            label_y = mid_y + perp_y * offset
        else:
            label_x = mid_x
            label_y = mid_y

        # Draw the label showing actual S value (0-1 range)
        ax.text(
            label_x, label_y,
            f"{weight:.3f}",
            fontsize=11,
            fontweight='bold',
            color='#2E3440',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8)
        )

    # Title and styling
    ax.set_title(
        f'S Matrix Topic Relationship Graph\n{table_name.replace("_", " ").title()}',
        fontsize=18,
        fontweight='bold',
        pad=20,
        color='#2E3440'
    )

    # Legend for edge weights
    legend_elements = [
        plt.Line2D([0], [0], color='#5E81AC', linewidth=5, alpha=1.0, label='Strong Connection'),
        plt.Line2D([0], [0], color='#5E81AC', linewidth=2.5, alpha=0.6, label='Medium Connection'),
        plt.Line2D([0], [0], color='#5E81AC', linewidth=1, alpha=0.3, label='Weak Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    # Add statistics text
    stats_text = (
        f"Topics: {n_topics}\n"
        f"Connections: {edges_added}\n"
        f"Layout: {layout.title()}\n"
        f"Threshold: {threshold:.3f}"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.axis('off')
    plt.tight_layout()

    # Save
    saved_path = None
    if output_path:
        filename = f"{table_name}_s_matrix_graph.png"
        file_path = output_path / filename
        plt.savefig(
            file_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none'
        )
        saved_path = str(file_path)
        print(f"💾 Network graph saved: {saved_path}")

    plt.close()

    return saved_path


def _create_custom_matplotlib_graph(
    s_matrix: np.ndarray,
    output_path: Optional[Path],
    table_name: str,
    threshold: float,
    layout: str,
    figsize: Tuple[int, int]
) -> Optional[str]:
    """Create custom matplotlib graph with smart label placement."""
    try:
        import networkx as nx
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("⚠️  NetworkX not installed. Skipping custom graph.")
        return None

    n_topics = s_matrix.shape[0]

    # Calculate total sum of S matrix for percentage calculation
    total_sum = np.sum(s_matrix)
    print(f"📊 Total S matrix sum: {total_sum:.2f}")

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes (topics)
    for i in range(n_topics):
        G.add_node(i, label=f"T{i+1}")

    # Add edges (connections between topics)
    edges_data = []
    for col in range(n_topics):  # Column = source topic
        for row in range(n_topics):  # Row = target topic
            weight = s_matrix[row, col]
            if weight > threshold and row != col:  # Skip self-connections
                percentage = (weight / total_sum) * 100 if total_sum > 0 else 0
                G.add_edge(col, row, weight=weight, percentage=percentage)
                edges_data.append((col, row, weight, percentage))

    if len(edges_data) == 0:
        print(f"⚠️  No edges above threshold {threshold}. Try lowering the threshold.")
        return None

    print(f"📈 Custom graph: {n_topics} nodes, {len(edges_data)} edges")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_aspect('equal')

    # Choose layout
    if layout == "hierarchical":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.circular_layout(G)

    # Get edge weights for normalization
    weights = [e[2] for e in edges_data]
    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 0

    # Node radius for calculating arrow start/end points
    node_radius = 0.05  # Reduced from 0.08

    # Draw edges with custom arrow patches and labels parallel to edges
    for col, row, weight, percentage in edges_data:
        x1, y1 = pos[col]
        x2, y2 = pos[row]

        # Calculate direction vector and shorten to stop at node edge
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0:
            # Unit vector
            ux = dx / dist
            uy = dy / dist

            # Adjust start and end points to stop at circle edge
            start_x = x1 + ux * node_radius
            start_y = y1 + uy * node_radius
            end_x = x2 - ux * node_radius
            end_y = y2 - uy * node_radius
        else:
            start_x, start_y = x1, y1
            end_x, end_y = x2, y2

        # Normalize weight for visualization
        normalized_weight = (weight - min_weight) / (max_weight - min_weight + 1e-10)
        edge_width = 1 + 4 * normalized_weight
        edge_alpha = 0.3 + 0.7 * normalized_weight

        # Draw edge with arrow (larger arrowhead for visibility)
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->',
            mutation_scale=25,  # Controls arrow size
            color='#5E81AC',
            linewidth=edge_width,
            alpha=edge_alpha,
            zorder=1,
            shrinkA=0,
            shrinkB=0
        )
        ax.add_patch(arrow)

        # Calculate edge length
        dx = x2 - x1
        dy = y2 - y1
        edge_length = np.sqrt(dx**2 + dy**2)

        # Only add label if edge is long enough
        if edge_length > 0.3:  # Skip labels on very short edges
            # Calculate angle of edge
            angle = np.degrees(np.arctan2(dy, dx))

            # Normalize angle to [-90, 90] for better readability
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # Label position at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Draw label rotated along edge direction showing actual S value (0-1 range)
            ax.text(
                mid_x, mid_y,
                f"{weight:.3f}",
                fontsize=9,
                fontweight='bold',
                color='#2E3440',
                ha='center',
                va='bottom',
                rotation=angle,
                rotation_mode='anchor',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85),
                zorder=3
            )

    # Draw nodes with smaller size
    node_colors = ['#88C0D0' for _ in range(n_topics)]
    for i, (node, (x, y)) in enumerate(pos.items()):
        circle = plt.Circle(
            (x, y), node_radius,  # Use the same radius as edge calculations
            color=node_colors[i],
            alpha=0.95,
            edgecolor='#2E3440',
            linewidth=2.5,
            zorder=4
        )
        ax.add_patch(circle)

        # Draw node label (smaller font for smaller circles)
        ax.text(
            x, y,
            f"T{node+1}",
            fontsize=9,
            fontweight='bold',
            color='white',
            ha='center',
            va='center',
            zorder=5
        )

    # Title and styling
    ax.set_title(
        f'S Matrix Topic Relationship Graph (Custom)\n{table_name.replace("_", " ").title()}',
        fontsize=18,
        fontweight='bold',
        pad=20,
        color='#2E3440'
    )

    # Add statistics text
    stats_text = (
        f"Topics: {n_topics}\n"
        f"Connections: {len(edges_data)}\n"
        f"Layout: {layout.title()}\n"
        f"Threshold: {threshold:.3f}"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.axis('off')
    ax.autoscale()
    ax.margins(0.1)
    plt.tight_layout()

    # Save
    saved_path = None
    if output_path:
        filename = f"{table_name}_s_matrix_graph_custom.png"
        file_path = output_path / filename
        plt.savefig(
            file_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none'
        )
        saved_path = str(file_path)
        print(f"💾 Custom graph saved: {saved_path}")

    plt.close()

    return saved_path


def _create_interactive_graph(
    s_matrix: np.ndarray,
    output_path: Optional[Path],
    table_name: str,
    threshold: float
) -> Optional[str]:
    """Create interactive network graph using plotly."""
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError:
        print("⚠️  Plotly or NetworkX not installed. Skipping interactive graph. Install with: pip install plotly networkx")
        return None

    n_topics = s_matrix.shape[0]

    # Calculate total sum of S matrix for percentage calculation
    total_sum = np.sum(s_matrix)

    # Create directed graph
    G = nx.DiGraph()
    for i in range(n_topics):
        G.add_node(i)

    edge_traces = []
    edge_info = []

    for col in range(n_topics):
        for row in range(n_topics):
            weight = s_matrix[row, col]
            if weight > threshold and row != col:
                percentage = (weight / total_sum) * 100 if total_sum > 0 else 0
                G.add_edge(col, row, weight=weight, percentage=percentage)
                edge_info.append((col, row, weight, percentage))

    if len(edge_info) == 0:
        print(f"⚠️  No edges above threshold {threshold} for interactive graph.")
        return None

    # Layout
    pos = nx.circular_layout(G)

    # Create edge traces
    for source, target, weight, percentage in edge_info:
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # Normalize weight for visualization
        max_weight = max([e[2] for e in edge_info])
        normalized_weight = weight / max_weight

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=1 + 4 * normalized_weight,
                color=f'rgba(94, 129, 172, {0.3 + 0.7 * normalized_weight})'
            ),
            hoverinfo='text',
            text=f'T{source+1} → T{target+1}<br>Strength: {weight:.2f}<br>Percentage: {percentage:.2f}%',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Calculate node statistics
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))
        in_strength = sum([e[2]['weight'] for e in in_edges])
        out_strength = sum([e[2]['weight'] for e in out_edges])

        node_text.append(
            f'<b>Topic {node+1}</b><br>'
            f'Incoming: {len(in_edges)} edges ({in_strength:.2f})<br>'
            f'Outgoing: {len(out_edges)} edges ({out_strength:.2f})'
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f'T{i+1}' for i in range(n_topics)],
        hovertext=node_text,
        textposition="middle center",
        textfont=dict(size=12, color='white', family='Arial Black'),
        marker=dict(
            size=40,
            color='#88C0D0',
            line=dict(width=2, color='#2E3440')
        ),
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(
            text=f'S Matrix Topic Relationship Graph (Interactive)<br>{table_name.replace("_", " ").title()}',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=900,
        height=900
    )

    # Save
    saved_path = None
    if output_path:
        filename = f"{table_name}_s_matrix_interactive.html"
        file_path = output_path / filename
        fig.write_html(file_path)
        saved_path = str(file_path)
        print(f"💾 Interactive graph saved: {saved_path}")

    return saved_path


def _create_s_matrix_heatmap(
    s_matrix: np.ndarray,
    output_path: Optional[Path],
    table_name: str,
    figsize: Tuple[int, int]
) -> Optional[str]:
    """Create heatmap visualization of the S matrix."""
    n_topics = s_matrix.shape[0]

    # Calculate total sum of S matrix for percentage calculation
    total_sum = np.sum(s_matrix)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Create heatmap
    im = ax.imshow(s_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(n_topics))
    ax.set_yticks(np.arange(n_topics))
    ax.set_xticklabels([f'T{i+1}' for i in range(n_topics)])
    ax.set_yticklabels([f'T{i+1}' for i in range(n_topics)])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Connection Strength', rotation=270, labelpad=20, fontsize=12)

    # Add text annotations for non-zero values with both absolute value and percentage
    threshold = np.max(s_matrix) * 0.01  # 1% of max
    for i in range(n_topics):
        for j in range(n_topics):
            value = s_matrix[i, j]
            if value > threshold:
                percentage = (value / total_sum) * 100 if total_sum > 0 else 0
                text_color = 'white' if value > np.max(s_matrix) * 0.5 else 'black'
                # Show S value (0-1 range) on first line, percentage on second line
                ax.text(j, i, f'{value:.3f}\n({percentage:.1f}%)',
                       ha="center", va="center", color=text_color, fontsize=7,
                       fontweight='bold')

    # Title and labels
    ax.set_title(
        f'S Matrix Heatmap - Topic Relationships\n{table_name.replace("_", " ").title()}',
        fontsize=16,
        fontweight='bold',
        pad=20,
        color='#2E3440'
    )
    ax.set_xlabel('Source Topic (Column)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Target Topic (Row)', fontsize=12, fontweight='medium')

    # Grid
    ax.set_xticks(np.arange(n_topics + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_topics + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()

    # Save
    saved_path = None
    if output_path:
        filename = f"{table_name}_s_matrix_heatmap.png"
        file_path = output_path / filename
        plt.savefig(
            file_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none'
        )
        saved_path = str(file_path)
        print(f"💾 Heatmap saved: {saved_path}")

    plt.close()

    return saved_path


def _print_s_matrix_statistics(s_matrix: np.ndarray, threshold: float):
    """Print summary statistics about the S matrix."""
    n_topics = s_matrix.shape[0]

    # Calculate total sum
    total_sum = np.sum(s_matrix)

    # Count non-zero connections
    non_zero_mask = s_matrix > threshold
    np.fill_diagonal(non_zero_mask, False)  # Exclude diagonal
    total_connections = np.sum(non_zero_mask)

    # Find strongest connections
    flat_indices = np.argsort(s_matrix.flatten())[::-1]
    top_connections = []

    for idx in flat_indices:
        row, col = np.unravel_index(idx, s_matrix.shape)
        value = s_matrix[row, col]
        if value > threshold and row != col and len(top_connections) < 5:
            percentage = (value / total_sum) * 100 if total_sum > 0 else 0
            top_connections.append((col, row, value, percentage))

    print(f"\n📊 S Matrix Statistics:")
    print(f"  • Matrix size: {n_topics}×{n_topics}")
    print(f"  • Total sum of all values: {total_sum:.2f}")
    print(f"  • Total connections (>{threshold}): {total_connections}")
    print(f"  • Connection density: {total_connections / (n_topics * (n_topics-1)) * 100:.1f}%")
    print(f"  • Max connection strength: {np.max(s_matrix):.2f}")
    print(f"  • Mean connection strength: {np.mean(s_matrix[non_zero_mask]):.2f}" if total_connections > 0 else "")

    if top_connections:
        print(f"\n🔝 Top 5 Strongest Connections:")
        for i, (source, target, strength, percentage) in enumerate(top_connections, 1):
            print(f"  {i}. T{source+1} → T{target+1}: {strength:.2f} ({percentage:.2f}% of total)")


def load_s_matrix_from_json(json_path: Union[str, Path]) -> np.ndarray:
    """
    Load S matrix from JSON file saved by save_s_matrix.py.

    Args:
        json_path: Path to the JSON file containing S matrix

    Returns:
        S matrix as numpy array

    Example:
        >>> s_matrix = load_s_matrix_from_json("output/my_analysis_s_matrix.json")
        >>> visualize_s_matrix_graph(s_matrix, ...)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return np.array(data['s_matrix'])
