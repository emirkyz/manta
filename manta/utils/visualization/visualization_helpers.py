"""
Shared visualization helper functions for MANTA topic modeling.

This module provides common utilities used across multiple visualization modules,
particularly for generating maximally distinct color palettes for topic visualization.
"""

from typing import List
import numpy as np


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
