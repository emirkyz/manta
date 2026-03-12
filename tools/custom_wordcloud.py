"""
Custom Shape Wordcloud Generator

Generate wordclouds from selected topics with support for custom mask images.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from wordcloud import WordCloud


def crop_and_resize_mask(mask: np.ndarray, target_size: int = None, padding: int = 20) -> np.ndarray:
    """
    Crop mask to bounding box of the shape and optionally resize.

    Args:
        mask: Binary mask array (0 = drawable, 255 = blocked)
        target_size: Target size for the largest dimension. If None, keeps original size.
        padding: Padding around the shape in pixels

    Returns:
        Cropped (and optionally resized) mask
    """
    # Find bounding box of the shape (where mask == 0)
    rows = np.any(mask == 0, axis=1)
    cols = np.any(mask == 0, axis=0)

    if not rows.any() or not cols.any():
        return mask  # No shape found

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    rmin = max(0, rmin - padding)
    rmax = min(mask.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(mask.shape[1], cmax + padding)

    # Crop
    cropped = mask[rmin:rmax, cmin:cmax]

    # Resize if target_size specified
    if target_size:
        from PIL import Image
        h, w = cropped.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img = Image.fromarray(cropped)
        img = img.resize((new_w, new_h), Image.NEAREST)
        cropped = np.array(img)

    return cropped


def load_mask_image(mask_path: str, invert: bool = False, threshold: int = None) -> np.ndarray:
    """
    Load and convert image to proper mask format for WordCloud.

    WordCloud expects: 255 = blocked (no words), lower values = drawable.

    For best results, use an image with:
    - WHITE background (will be blocked)
    - BLACK shape where you want words

    Args:
        mask_path: Path to mask image (PNG, JPG, etc.)
        invert: If True, invert the mask (use when shape is white on black)
        threshold: Grayscale threshold to separate shape from background.
                   Pixels below threshold = shape (drawable).
                   If None, auto-detects based on image.

    Returns:
        numpy array suitable for WordCloud mask parameter
    """
    img = Image.open(mask_path)

    # Handle PNG with transparency (RGBA)
    if img.mode == 'RGBA':
        # Check if alpha channel has meaningful variation (true transparency)
        alpha = np.array(img.split()[3])
        alpha_std = np.std(alpha)

        if alpha_std > 50:  # Has meaningful transparency
            # Transparent = blocked (255), opaque = drawable (0)
            mask = np.where(alpha > 128, 0, 255).astype(np.uint8)
            if invert:
                mask = 255 - mask
            return mask

    # Convert to grayscale
    img_gray = img.convert('L')
    gray = np.array(img_gray)

    # Auto-detect threshold if not provided
    # Find the darkest region (the shape) vs background
    if threshold is None:
        # Use Otsu-like approach: find value that separates dark shape from background
        min_val = gray.min()
        max_val = gray.max()
        # Threshold at 1/3 between min and max (assumes dark shape on lighter background)
        threshold = min_val + (max_val - min_val) // 3

    # Create binary mask: below threshold = drawable (0), above = blocked (255)
    mask = np.where(gray <= threshold, 0, 255).astype(np.uint8)

    if invert:
        mask = 255 - mask

    return mask


def create_default_heart_mask(size: int = 800) -> np.ndarray:
    """
    Create a heart-shaped mask using parametric equations.

    Args:
        size: Image dimension (square output of size x size)

    Returns:
        numpy array where low values = inside heart (drawable), 255 = outside (blocked)
    """
    x = np.linspace(-1.3, 1.3, size)
    y = np.linspace(-1.5, 1.1, size)
    X, Y = np.meshgrid(x, y)

    # Heart curve equation: (x^2 + y^2 - 1)^3 - x^2 * y^3 <= 0
    heart = (X**2 + Y**2 - 1)**3 - X**2 * Y**3

    # Create mask: 0 inside heart (drawable), 255 outside (blocked)
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[heart <= 0] = 0

    # Flip vertically so heart points up
    mask = np.flipud(mask)

    return mask


def load_relevance_json(json_path: str) -> dict:
    """
    Load and parse relevance_top_words.json file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary containing topic word scores

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON structure is invalid
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'relevance' not in data:
        raise ValueError("JSON missing 'relevance' key. Expected format: {'relevance': {'topic_01': {...}, ...}}")

    return data['relevance']


def combine_topic_words(
    relevance_data: dict,
    topic_ids: list,
    method: str = 'sum'
) -> dict:
    """
    Combine word scores from multiple topics.

    Args:
        relevance_data: Dictionary from load_relevance_json()
        topic_ids: List of topic IDs (e.g., ['topic_01', 'topic_02'])
        method: Combination method - 'sum', 'average', or 'max'

    Returns:
        Combined word scores dictionary
    """
    combined = {}

    for topic_id in topic_ids:
        if topic_id not in relevance_data:
            print(f"Warning: {topic_id} not found in data, skipping")
            continue

        for word, score in relevance_data[topic_id].items():
            if word not in combined:
                combined[word] = []
            combined[word].append(score)

    # Apply combination method
    result = {}
    for word, scores in combined.items():
        if method == 'sum':
            result[word] = sum(scores)
        elif method == 'average':
            result[word] = sum(scores) / len(scores)
        elif method == 'max':
            result[word] = max(scores)  # Less negative = more relevant
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sum', 'average', or 'max'")

    return result


def normalize_scores(word_scores: dict) -> dict:
    """
    Transform relevance scores to positive frequencies for wordcloud.

    Relevance scores are typically negative log values where
    higher (less negative) means more relevant.

    Args:
        word_scores: Raw combined scores (negative values)

    Returns:
        Positive frequencies suitable for WordCloud
    """
    if not word_scores:
        return {}

    min_score = min(word_scores.values())

    # Shift so minimum becomes 1, ensuring all positive
    normalized = {}
    for word, score in word_scores.items():
        normalized[word] = score - min_score + 1.0

    return normalized


def generate_custom_wordcloud(
    json_path: str,
    topic_ids: list,
    output_path: str = "custom_wordcloud.png",
    mask_path: str = None,
    method: str = "sum",
    size: int = 800,
    colormap: str = "Reds",
    background_color: str = "white",
    contour_color: str = "red",
    contour_width: int = 2,
    max_words: int = 200,
    invert_mask: bool = False,
    mask_threshold: int = None,
    crop_mask: bool = True,
    mask_padding: int = 20
) -> None:
    """
    Generate wordcloud from selected topics with optional custom mask.

    Args:
        json_path: Path to relevance_top_words.json
        topic_ids: List of topic IDs (e.g., ['topic_01', 'topic_02'])
        output_path: Where to save the output image
        mask_path: Optional path to mask image (black shape on white background works best)
        method: How to combine scores - 'sum', 'average', or 'max'
        size: Image size if no mask provided (generates default heart). Also used for resizing cropped mask.
        colormap: Matplotlib colormap for word colors (e.g., 'Reds', 'Blues', 'viridis')
        background_color: Background color
        contour_color: Outline color around the shape
        contour_width: Outline thickness (0 to disable)
        max_words: Maximum number of words to display
        invert_mask: If True, invert the mask (for white shape on black background)
        mask_threshold: Grayscale threshold (0-255) for mask. Pixels darker than this = shape.
                        Lower values = stricter (only darkest pixels). Default auto-detects.
        crop_mask: If True, crop mask to shape's bounding box and resize to fill image (default: True)
        mask_padding: Padding around cropped shape in pixels (default: 20)
    """
    # Load data
    print(f"Loading: {json_path}")
    relevance_data = load_relevance_json(json_path)

    available_topics = list(relevance_data.keys())
    print(f"Available topics: {available_topics}")

    # Combine words from selected topics
    print(f"Combining topics: {topic_ids} using '{method}' method")
    combined_scores = combine_topic_words(relevance_data, topic_ids, method)

    if not combined_scores:
        raise ValueError("No words found in selected topics")

    print(f"Total unique words: {len(combined_scores)}")

    # Normalize to positive frequencies
    frequencies = normalize_scores(combined_scores)

    # Load or create mask
    if mask_path:
        print(f"Loading mask from: {mask_path}")
        mask = load_mask_image(mask_path, invert=invert_mask, threshold=mask_threshold)

        if crop_mask:
            print(f"Cropping and resizing mask to {size}px")
            mask = crop_and_resize_mask(mask, target_size=size, padding=mask_padding)

        width, height = mask.shape[1], mask.shape[0]
    else:
        print(f"Creating default heart mask ({size}x{size})")
        mask = create_default_heart_mask(size)
        width, height = size, size

    # Create wordcloud
    wc = WordCloud(
        mask=mask,
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        contour_width=contour_width,
        contour_color=contour_color,
        max_words=max_words,
        min_font_size=8,
        prefer_horizontal=0.7
    )

    wc.generate_from_frequencies(frequencies)

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = wc.to_image()
    image.save(str(output), dpi=(300, 300))

    print(f"Wordcloud saved to: {output}")


# Convenience function for quick heart wordcloud
def generate_heart_wordcloud(
    json_path: str,
    topic_ids: list,
    output_path: str = "heart_wordcloud.png",
    **kwargs
) -> None:
    """
    Convenience function for generating heart-shaped wordcloud.

    Same as generate_custom_wordcloud but with heart-optimized defaults.
    """
    defaults = {
        'colormap': 'Reds',
        'contour_color': 'darkred',
        'contour_width': 3
    }
    defaults.update(kwargs)

    generate_custom_wordcloud(
        json_path=json_path,
        topic_ids=topic_ids,
        output_path=output_path,
        **defaults
    )
