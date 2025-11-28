"""
MANTA LDAvis-Style Interactive Visualization

This module creates pyLDAvis-inspired interactive visualizations for MANTA NMF topic modeling results.
It transforms NMF outputs (W, H matrices) into an interactive HTML visualization featuring:
- Topic bubble chart with inter-topic distances
- Interactive term frequency bars
- Dynamic topic exploration interface

Author: MANTA Team
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from scipy.stats import entropy
from sklearn.manifold import MDS


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


def _reorder_H_by_pairing(H: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """
    Reorder H matrix rows based on topic pairing from S matrix.

    This ensures that H rows are reordered to match the topic ordering
    determined by the S matrix pairing (sorted by coupling strength).

    Args:
        H: Topic-word matrix (n_topics, n_words)
        s_matrix: S matrix from NMTF (k x k)

    Returns:
        H_reordered: H matrix with rows reordered by topic pairing
    """
    ind, _ = _sort_matrices(s_matrix)
    n_topics = H.shape[0]
    H_reordered = np.zeros_like(H)

    for new_idx, (word_cluster_id, doc_cluster_id) in enumerate(ind):
        if new_idx < n_topics:
            H_reordered[new_idx, :] = H[word_cluster_id, :]

    return H_reordered


def create_manta_ldavis(w_matrix: np.ndarray,
                        h_matrix: np.ndarray,
                        s_matrix: Optional[np.ndarray] = None,
                        vocab: List[str] = None,
                        doc_lengths: Optional[List[int]] = None,
                        term_frequency: Optional[List[int]] = None,
                        output_dir: Optional[Union[str, Path]] = None,
                        table_name: str = "manta_ldavis",
                        lambda_step: float = 0.01,
                        plot_opts: Dict[str, Any] = None,
                        sort_topics: bool = True,
                        tokenizer = None,
                        emoji_map = None) -> Optional[str]:
    """
    Create an interactive LDAvis-style visualization for MANTA NMF results.
    
    Args:
        w_matrix: Document-topic matrix (n_docs x n_topics)
        h_matrix: Topic-word matrix (n_topics x n_vocab)
        vocab: List of vocabulary words
        doc_lengths: List of document lengths (optional)
        term_frequency: List of term frequencies (optional)
        output_dir: Directory to save HTML file
        table_name: Base name for output file
        lambda_step: Step size for lambda slider
        plot_opts: Additional plotting options
        sort_topics: Whether to sort topics by size
        
    Returns:
        Path to saved HTML file or None if failed
    """
    try:
        print("🎨 Creating MANTA LDAvis visualization...")
        
        # Input validation
        if w_matrix is None or h_matrix is None:
            raise ValueError("Both W and H matrices must be provided")
        
        # Ensure matrices have compatible dimensions
        if hasattr(w_matrix, 'toarray'):
            w_shape = w_matrix.shape
        else:
            w_shape = np.asarray(w_matrix).shape
            
        if hasattr(h_matrix, 'toarray'):
            h_shape = h_matrix.shape
        else:
            h_shape = np.asarray(h_matrix).shape
            
        if len(w_shape) != 2 or len(h_shape) != 2:
            raise ValueError("W and H matrices must be 2-dimensional")
            
        if w_shape[1] != h_shape[0]:
            raise ValueError(f"Matrix dimension mismatch: W has {w_shape[1]} topics, H has {h_shape[0]} topics")
        
        print(f"📊 Processing {w_shape[0]:,} documents, {h_shape[0]} topics, {h_shape[1]:,} vocabulary terms")
        
        # Prepare data for visualization
        vis_data = prepare_manta_data(
            w_matrix=w_matrix,
            h_matrix=h_matrix,
            vocab=vocab,
            doc_lengths=doc_lengths,
            term_frequency=term_frequency,
            lambda_step=lambda_step,
            sort_topics=sort_topics,
            tokenizer=tokenizer,
            emoji_map=emoji_map,
            s_matrix=s_matrix
        )
        
        # Generate HTML visualization
        html_content = generate_html_visualization(vis_data, plot_opts or {})
        
        # Save to file
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"{table_name}_interactive_ldavis.html"
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"💾 Interactive LDAvis visualization saved to: {file_path}")
            return str(file_path)
        
        return html_content
        
    except ValueError as e:
        print(f"❌ Input validation error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error creating MANTA LDAvis visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_manta_data(w_matrix: np.ndarray,
                       h_matrix: np.ndarray,
                       vocab: List[str] = None,
                       doc_lengths: Optional[List[int]] = None,
                       term_frequency: Optional[List[int]] = None,
                       lambda_step: float = 0.01,
                       sort_topics: bool = True,
                       tokenizer = None,
                       emoji_map = None,
                       s_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Prepare MANTA NMF data for LDAvis-style visualization.

    Args:
        w_matrix: Document-topic matrix (n_docs x n_topics)
        h_matrix: Topic-word matrix (n_topics x n_vocab)
        vocab: List of vocabulary words
        doc_lengths: List of document lengths (optional)
        term_frequency: List of term frequencies (optional)
        lambda_step: Step size for lambda slider
        sort_topics: Whether to sort topics by size
        tokenizer: Tokenizer for vocabulary creation (optional)
        emoji_map: Emoji mapping for vocabulary (optional)
        s_matrix: S matrix for NMTF (n_topics x n_topics, optional).
                 Expected to be L1 column-normalized for consistent interpretation.

    Returns:
        Dictionary containing all data needed for visualization
    """
    print("🔄 Preparing data for LDAvis visualization...")
    
    # Input validation
    if tokenizer is None and (vocab is None or len(vocab) == 0):
        raise ValueError("Either tokenizer or vocabulary must be provided")
    
    # Additional validation for numerical stability
    if np.any(np.isnan(w_matrix)) or np.any(np.isnan(h_matrix)):
        raise ValueError("Input matrices contain NaN values")
        
    if np.any(np.isinf(w_matrix)) or np.any(np.isinf(h_matrix)):
        raise ValueError("Input matrices contain infinite values")
    
    # Ensure matrices are dense numpy arrays
    if hasattr(w_matrix, 'toarray'):
        w_matrix = w_matrix.toarray()
    if hasattr(h_matrix, 'toarray'):
        h_matrix = h_matrix.toarray()

    w_matrix = np.asarray(w_matrix)
    h_matrix = np.asarray(h_matrix)

    n_docs, n_topics = w_matrix.shape
    n_topics_h, n_vocab = h_matrix.shape

    assert n_topics == n_topics_h, f"Matrix dimension mismatch: W has {n_topics} topics, H has {n_topics_h}"

    # For NMTF models, reorder W and H based on topic pairing from S matrix
    if s_matrix is not None:
        w_matrix = _reorder_W_by_pairing(w_matrix, s_matrix)
        h_matrix = _reorder_H_by_pairing(h_matrix, s_matrix)
        print("🔄 Reordered W and H matrices based on NMTF topic pairing")
    
    # Create vocabulary from tokenizer if needed
    if vocab is None and tokenizer is not None:
        print("Creating vocabulary from tokenizer...")
        vocab = _create_vocab_from_tokenizer(tokenizer, n_vocab, emoji_map)
    
    if vocab is None or len(vocab) != n_vocab:
        raise ValueError(f"Vocabulary size mismatch: H has {n_vocab} terms, vocab has {len(vocab) if vocab else 0}")
    
    # Normalize matrices
    w_matrix_norm = w_matrix / (w_matrix.sum(axis=1, keepdims=True) + 1e-10)
    h_matrix_norm = h_matrix / (h_matrix.sum(axis=1, keepdims=True) + 1e-10)

    # Calculate topic sizes (total probability mass)
    # Note: For NMTF, w_matrix has already been reordered by topic pairing above
    from ...utils.analysis import get_dominant_topics
    dominant_topics = get_dominant_topics(w_matrix, min_score=0.0)
    valid_mask = dominant_topics != -1
    n_topics_effective = n_topics  # Use the actual number of topics
    topic_sizes = np.zeros(n_topics_effective)
    for topic_idx in range(n_topics_effective):
        topic_sizes[topic_idx] = np.sum(dominant_topics == topic_idx)
    
    # Calculate term frequencies if not provided
    # Use both H matrix and W matrix to get proper corpus-wide term frequencies
    if term_frequency is None:
        # Method 1: Direct from H matrix (topic-word weights)
        # Method 2: Weight by topic sizes for corpus representation
        topic_weights = w_matrix.sum(axis=0)  # Total weight per topic
        # Weighted term frequency: sum over topics of (topic_weight * word_prob_in_topic)
        term_frequency = np.sum(h_matrix * topic_weights.reshape(-1, 1), axis=0)
    
    # Calculate document lengths if not provided  
    if doc_lengths is None:
        doc_lengths = [100] * n_docs  # Default assumption
    
    # Calculate inter-topic distances using Jensen-Shannon divergence
    topic_distances = calculate_topic_distances(h_matrix_norm)
    
    # Project topics to 2D using MDS
    topic_coordinates = project_topics_to_2d(topic_distances)
    
    # Calculate term-topic frequencies
    term_topic_freq = calculate_term_topic_frequencies(h_matrix, w_matrix)
    
    # Prepare topic info
    topic_info = prepare_topic_info(
        h_matrix_norm, vocab, topic_sizes, term_frequency, 
        term_topic_freq, lambda_step
    )
    
    # Sort topics by size if requested
    if sort_topics:
        sort_idx = np.argsort(topic_sizes)[::-1]  # Descending order
        
        # Reorder coordinates and sizes
        topic_coordinates = topic_coordinates[sort_idx]
        topic_sizes = topic_sizes[sort_idx]
        
        # Fix topic_info reordering - need to remap topic categories properly
        topic_info_reordered = []
        
        # Keep 'Default' category unchanged
        default_info = topic_info[topic_info['Category'] == 'Default'].copy()
        topic_info_reordered.append(default_info)
        
        # Reorder topic-specific entries and renumber them
        for new_idx, orig_idx in enumerate(sort_idx):
            orig_category = f'Topic{orig_idx + 1}'
            topic_entries = topic_info[topic_info['Category'] == orig_category].copy()
            # Renumber the category to maintain 1-based indexing in display order
            topic_entries['Category'] = f'Topic{new_idx + 1}'
            topic_info_reordered.append(topic_entries)
        
        topic_info = pd.concat(topic_info_reordered, ignore_index=True)
    
    # Prepare the final data structure
    vis_data = {
        'topic_coordinates': topic_coordinates.tolist(),
        'topic_info': topic_info.to_dict('records'),
        'token_table': prepare_token_table(h_matrix_norm, vocab, term_topic_freq).to_dict('records'),
        'R': min(30, len(vocab)),  # Number of terms to show
        'lambda_step': lambda_step,
        'plot_opts': {
            'xlab': 'PC1',
            'ylab': 'PC2'
        },
        'topic_order': list(range(1, n_topics + 1)),
        'client_topic_order': list(range(1, n_topics + 1)),
        'marginal_topic_dist': (topic_sizes / topic_sizes.sum()).tolist(),
        'topic_sizes': topic_sizes.tolist(),
        'vocab': vocab,
        'term_frequency': term_frequency.tolist() if hasattr(term_frequency, 'tolist') else list(term_frequency)
    }
    
    return vis_data


def calculate_topic_distances(topic_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate inter-topic distances using Jensen-Shannon divergence.
    
    Uses an optimized vectorized approach and proper normalization for better
    interpretability of topic relationships.
    
    Args:
        topic_matrix: Normalized topic-word matrix (n_topics x n_vocab)
        
    Returns:
        Symmetric distance matrix (n_topics x n_topics)
    """
    n_topics = topic_matrix.shape[0]
    
    # Add small epsilon to avoid log(0) and ensure numerical stability
    epsilon = 1e-12
    topic_matrix_safe = topic_matrix + epsilon
    
    # Renormalize after adding epsilon
    topic_matrix_safe = topic_matrix_safe / topic_matrix_safe.sum(axis=1, keepdims=True)
    
    # Initialize symmetric distance matrix
    distances = np.zeros((n_topics, n_topics))
    
    # Calculate Jensen-Shannon divergence for upper triangle only (optimization)
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            p = topic_matrix_safe[i]
            q = topic_matrix_safe[j]
            m = 0.5 * (p + q)
            
            # Calculate JS divergence with proper base-2 logarithm for interpretability
            js_div = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
            
            # JS distance is sqrt of JS divergence, bounded between 0 and 1
            js_distance = np.sqrt(np.clip(js_div, 0, 1))
            
            # Fill both upper and lower triangle (symmetric matrix)
            distances[i, j] = js_distance
            distances[j, i] = js_distance
    
    return distances


def project_topics_to_2d(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Project topics to 2D space using multidimensional scaling with overlap prevention.
    
    Args:
        distance_matrix: Topic distance matrix
        
    Returns:
        2D coordinates for topics (n_topics x 2)
    """
    # Use MDS to project to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)
    
    # Apply force-directed layout to prevent overlapping
    coordinates = _apply_force_layout(coordinates, distance_matrix)
    
    return coordinates


def _apply_force_layout(coordinates: np.ndarray, distance_matrix: np.ndarray, 
                       iterations: int = 100, repulsion_strength: float = 1.0) -> np.ndarray:
    """
    Apply force-directed layout to prevent overlapping circles.
    
    Args:
        coordinates: Initial 2D coordinates
        distance_matrix: Distance matrix for topics
        iterations: Number of iterations to run
        repulsion_strength: Strength of repulsion force
        
    Returns:
        Adjusted coordinates with minimal overlap
    """
    coords = coordinates.copy()
    n_topics = len(coords)
    
    for _ in range(iterations):
        forces = np.zeros_like(coords)
        
        for i in range(n_topics):
            for j in range(n_topics):
                if i != j:
                    # Calculate distance between topics
                    diff = coords[i] - coords[j]
                    dist = np.linalg.norm(diff)
                    
                    # Apply repulsion force if too close
                    min_distance = 0.3  # Minimum distance to maintain
                    if dist < min_distance and dist > 0:
                        force_magnitude = repulsion_strength * (min_distance - dist) / dist
                        forces[i] += force_magnitude * diff
        
        # Apply forces with damping
        coords += forces * 0.1
    
    return coords


def calculate_term_topic_frequencies(h_matrix: np.ndarray, w_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate term frequencies within each topic.
    
    Args:
        h_matrix: Topic-word matrix
        w_matrix: Document-topic matrix
        
    Returns:
        Term-topic frequency matrix
    """
    # Multiply H by topic weights from W to get actual term-topic frequencies
    topic_weights = w_matrix.sum(axis=0)  # Total weight per topic
    term_topic_freq = h_matrix * topic_weights.reshape(-1, 1)
    
    return term_topic_freq


def prepare_topic_info(h_matrix: np.ndarray, vocab: List[str], topic_sizes: np.ndarray,
                       term_frequency: np.ndarray, term_topic_freq: np.ndarray,
                       lambda_step: float = 0.01) -> pd.DataFrame:
    """
    Prepare topic information DataFrame for visualization.
    
    Args:
        h_matrix: Normalized topic-word matrix
        vocab: Vocabulary list
        topic_sizes: Topic size array
        term_frequency: Global term frequency
        term_topic_freq: Term-topic frequency matrix
        lambda_step: Lambda step size
        
    Returns:
        DataFrame with topic information
    """
    topic_info_list = []
    
    # Add overall term frequencies
    for i, term in enumerate(vocab):
        topic_info_list.append({
            'Category': 'Default',
            'Freq': float(term_frequency[i]),
            'Term': term,
            'Total': float(term_frequency[i]),
            'loglift': 0.0,
            'logprob': np.log(term_frequency[i] / term_frequency.sum())
        })
    
    # Add term frequencies for each topic using CORRECT probability normalization
    for topic_idx in range(h_matrix.shape[0]):
        # Extract the topic-word vector for this topic
        topic_word_vector = h_matrix[topic_idx]
        topic_term_freq = term_topic_freq[topic_idx]
        
        # CORRECT NORMALIZATION: Same as calculate_term_relevance function
        # Normalize to get proper probabilities (this is the key fix)
        topic_word_prob = topic_word_vector / (topic_word_vector.sum() + 1e-10)
        overall_word_prob = term_frequency / (term_frequency.sum() + 1e-10)
        
        # Calculate lift: ratio of word probability in topic to overall probability
        lift = topic_word_prob / (overall_word_prob + 1e-10)
        lift = np.clip(lift, 1e-10, None)  # Prevent negative or zero lift
        
        # CORRECT LOG CALCULATION: Use normalized probabilities
        logprob = np.log(topic_word_prob + 1e-10)
        loglift = np.log(lift)
        
        # Include ALL terms (not just top 100) for consistency with calculate_term_relevance
        # Filter will be applied later during visualization
        for word_idx in range(len(vocab)):
            if topic_word_vector[word_idx] > 1e-10:  # Only meaningful words
                term = vocab[word_idx]
                topic_info_list.append({
                    'Category': f'Topic{topic_idx + 1}',
                    'Freq': float(topic_term_freq[word_idx]),
                    'Term': term,
                    'Total': float(term_frequency[word_idx]),
                    'loglift': float(loglift[word_idx]),
                    'logprob': float(logprob[word_idx])
                })
    
    return pd.DataFrame(topic_info_list)


def prepare_token_table(h_matrix: np.ndarray, vocab: List[str], 
                        term_topic_freq: np.ndarray) -> pd.DataFrame:
    """
    Prepare token table for the visualization.
    
    Args:
        h_matrix: Normalized topic-word matrix
        vocab: Vocabulary list  
        term_topic_freq: Term-topic frequency matrix
        
    Returns:
        DataFrame with token information
    """
    token_list = []
    
    for term_idx, term in enumerate(vocab):
        for topic_idx in range(h_matrix.shape[0]):
            if term_topic_freq[topic_idx, term_idx] > 0:
                token_list.append({
                    'Term': term,
                    'Topic': topic_idx + 1,
                    'Freq': float(term_topic_freq[topic_idx, term_idx])
                })
    
    return pd.DataFrame(token_list)


def generate_html_visualization(vis_data: Dict[str, Any], 
                                plot_opts: Dict[str, Any] = None) -> str:
    """
    Generate HTML content for the interactive visualization.
    
    Args:
        vis_data: Prepared visualization data
        plot_opts: Additional plotting options
        
    Returns:
        HTML content string
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>MANTA Interactive Topic Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {
                font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
                margin: 0;
                padding: 24px;
                background: #f8fafc;
                min-height: 100vh;
                color: #2d3748;
            }
            
            .header {
                text-align: center;
                margin-bottom: 32px;
                padding: 32px 24px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #e2e8f0;
            }
            
            .header h1 {
                color: #1a202c;
                margin: 0 0 12px 0;
                font-size: 2.2em;
                font-weight: 600;
            }
            
            .header p {
                color: #64748b;
                margin: 0;
                font-size: 1.0em;
                font-weight: 400;
            }
            
            .container {
                display: flex;
                gap: 24px;
                max-width: 1600px;
                margin: 0 auto;
            }
            
            .left-panel, .right-panel {
                flex: 1;
                background: white;
                border-radius: 8px;
                padding: 28px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #e2e8f0;
            }
            
            .panel-title {
                font-size: 1.4em;
                font-weight: 600;
                color: #1a202c;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .controls {
                margin-bottom: 24px;
                text-align: center;
                padding: 16px;
                background: #f8fafc;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
            }
            
            .lambda-slider {
                width: 300px;
                height: 4px;
                margin: 10px;
                background: #e2e8f0;
                border-radius: 2px;
                outline: none;
                -webkit-appearance: none;
            }
            
            .lambda-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #3b82f6;
                cursor: pointer;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            }
            
            .lambda-label {
                font-size: 14px;
                color: #64748b;
                margin-bottom: 10px;
                font-weight: 500;
            }
            
            .topic-circle {
                cursor: pointer;
                stroke: #ffffff;
                stroke-width: 2px;
                transition: stroke-width 0.2s ease;
            }
            
            .topic-circle:hover {
                stroke: #374151;
                stroke-width: 3px;
            }
            
            .topic-circle.selected {
                stroke: #dc2626;
                stroke-width: 3px;
            }
            
            .topic-label {
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                font-weight: 700;
                font-size: 13px;
            }
            
            .term-bar {
                cursor: pointer;
                transition: opacity 0.2s ease;
            }
            
            .term-bar:hover {
                opacity: 0.8;
            }
            
            .axis {
                font-size: 12px;
                color: #4a5568;
            }
            
            .axis path,
            .axis line {
                fill: none;
                stroke: #cbd5e0;
                stroke-width: 1px;
                shape-rendering: crispEdges;
            }
            
            .tooltip {
                position: absolute;
                text-align: left;
                padding: 12px 16px;
                font-size: 12px;
                background: rgba(26, 32, 44, 0.95);
                color: white;
                border-radius: 6px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s ease;
                box-shadow: 0 2px 12px rgba(0,0,0,0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                max-width: 200px;
                line-height: 1.4;
            }
            
            .legend {
                font-size: 13px;
                margin-top: 24px;
                padding: 16px;
                background: rgba(247, 250, 252, 0.8);
                border-radius: 12px;
                border: 1px solid rgba(226, 232, 240, 0.8);
            }
            
            .legend-item {
                margin-bottom: 8px;
                padding: 4px 8px;
                border-radius: 6px;
                transition: background-color 0.2s ease;
                display: flex;
                align-items: center;
            }
            
            .legend-item:hover {
                background: rgba(255, 255, 255, 0.6);
            }
            
            .legend-color {
                width: 16px;
                height: 16px;
                margin-right: 12px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            }
            
            /* Responsive design */
            @media (max-width: 1200px) {
                .container {
                    flex-direction: column;
                }
                
                .lambda-slider {
                    width: 250px;
                }
            }
            
            @media (max-width: 768px) {
                body {
                    padding: 16px;
                }
                
                .header {
                    padding: 24px 16px;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .left-panel, .right-panel {
                    padding: 20px;
                }
                
                .lambda-slider {
                    width: 200px;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MANTA Interactive Topic Visualization</h1>
            <p>Explore topic relationships and term distributions in your document corpus</p>
        </div>
        
        <div class="container">
            <div class="left-panel">
                <div class="panel-title">Intertopic Distance Map</div>
                <p style="font-size: 13px; color: #64748b; text-align: center; margin-bottom: 16px;">
                    <em>Topics positioned by similarity - closer topics share more vocabulary</em>
                </p>
                <div id="topic-chart"></div>
                <div class="legend" id="topic-legend"></div>
            </div>
            
            <div class="right-panel">
                <div class="panel-title">Top Terms</div>
                <div class="controls">
                    <div class="lambda-label">Relevance (λ = <span id="lambda-value">0.6</span>)</div>
                    <input type="range" id="lambda-slider" class="lambda-slider" 
                           min="0" max="1" step="0.01" value="0.6">
                    <div style="font-size: 11px; color: #64748b; margin-top: 8px; text-align: center;">
                        λ=0: Most frequent in topic | λ=1: Most topic-specific | λ=0.6: Balanced
                    </div>
                </div>
                <div style="background: #f8fafc; padding: 12px; margin-bottom: 16px; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 12px; color: #4a5568; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 16px; height: 12px; background: #3b82f6; margin-right: 8px; border-radius: 2px;"></div>
                            <span>Topic frequency</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 16px; height: 12px; background: #e2e8f0; margin-right: 8px; border-radius: 2px;"></div>
                            <span>Corpus frequency</span>
                        </div>
                    </div>
                    <div style="font-size: 10px; color: #6b7280; margin-top: 6px; text-align: center;">
                        Hover bars for details • Click terms to copy • Blue extends beyond grey = topic-specific
                    </div>
                </div>
                <div id="term-chart"></div>
            </div>
        </div>
        
        <!-- Tooltip -->
        <div class="tooltip" id="tooltip"></div>
        
        <script>
            // Visualization data
            const visData = """ + json.dumps(vis_data, indent=2) + """;
            
            // Global state
            let selectedTopic = null;
            let currentLambda = 0.6;
            
            // Subtle professional color scale
            const colorScale = d3.scaleOrdinal([
                "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899",
                "#f43f5e", "#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16",
                "#22c55e", "#10b981", "#14b8a6", "#06b6d4"
            ]);
            
            // Initialize visualization
            initTopicChart();
            initTermChart();
            initControls();
            updateTermChart(); // Show default terms
            
            function initTopicChart() {
                const margin = {top: 30, right: 30, bottom: 50, left: 50};
                const width = 600 - margin.left - margin.right;
                const height = 500 - margin.top - margin.bottom;
                
                const svg = d3.select("#topic-chart")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom);
                
                const g = svg.append("g")
                    .attr("transform", `translate(${margin.left},${margin.top})`);
                
                // Extract coordinates and sizes
                const coordinates = visData.topic_coordinates;
                const sizes = visData.topic_sizes;
                const maxSize = Math.max(...sizes);
                
                // Scales with better padding
                const xExtent = d3.extent(coordinates, d => d[0]);
                const yExtent = d3.extent(coordinates, d => d[1]);
                
                // Add padding to the domains to prevent clipping
                const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
                const yPadding = (yExtent[1] - yExtent[0]) * 0.1;
                
                const xScale = d3.scaleLinear()
                    .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
                    .range([0, width]);
                
                const yScale = d3.scaleLinear()
                    .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
                    .range([height, 0]);
                
                const radiusScale = d3.scaleSqrt()
                    .domain([0, maxSize])
                    .range([10, 30]); // Better balance for visibility and positioning
                
                // Add grid lines for better positioning reference
                const xTicks = xScale.ticks(8);
                const yTicks = yScale.ticks(6);
                
                // Vertical grid lines
                g.selectAll(".grid-line-v")
                    .data(xTicks)
                    .enter()
                    .append("line")
                    .attr("class", "grid-line-v")
                    .attr("x1", d => xScale(d))
                    .attr("x2", d => xScale(d))
                    .attr("y1", 0)
                    .attr("y2", height)
                    .attr("stroke", "#f1f5f9")
                    .attr("stroke-width", 1);
                
                // Horizontal grid lines
                g.selectAll(".grid-line-h")
                    .data(yTicks)
                    .enter()
                    .append("line")
                    .attr("class", "grid-line-h")
                    .attr("x1", 0)
                    .attr("x2", width)
                    .attr("y1", d => yScale(d))
                    .attr("y2", d => yScale(d))
                    .attr("stroke", "#f1f5f9")
                    .attr("stroke-width", 1);
                
                // Add axes
                g.append("g")
                    .attr("class", "axis")
                    .attr("transform", `translate(0,${height})`)
                    .call(d3.axisBottom(xScale).ticks(8));
                
                g.append("g")
                    .attr("class", "axis")
                    .call(d3.axisLeft(yScale).ticks(6));
                
                // Add axis labels
                g.append("text")
                    .attr("class", "axis-label")
                    .attr("text-anchor", "middle")
                    .attr("x", width / 2)
                    .attr("y", height + 35)
                    .style("font-size", "14px")
                    .text("PC1");
                
                g.append("text")
                    .attr("class", "axis-label")
                    .attr("text-anchor", "middle")
                    .attr("transform", "rotate(-90)")
                    .attr("x", -height / 2)
                    .attr("y", -30)
                    .style("font-size", "14px")
                    .text("PC2");
                
                // Add topic circles
                const circles = g.selectAll(".topic-circle")
                    .data(coordinates)
                    .enter()
                    .append("circle")
                    .attr("class", "topic-circle")
                    .attr("cx", d => xScale(d[0]))
                    .attr("cy", d => yScale(d[1]))
                    .attr("r", (d, i) => radiusScale(sizes[i]))
                    .attr("fill", (d, i) => colorScale(i))
                    .attr("opacity", 0.7)
                    .on("click", function(event, d, i) {
                        // Handle topic selection
                        const topicIndex = coordinates.indexOf(d);
                        selectTopic(topicIndex);
                    })
                    .on("mouseover", function(event, d) {
                        const topicIndex = coordinates.indexOf(d);
                        const coord = coordinates[topicIndex];
                        const tooltipText = `
                            <strong>Topic ${topicIndex + 1}</strong><br>
                            Size: ${Math.round(sizes[topicIndex])} docs<br>
                            Position: (${coord[0].toFixed(2)}, ${coord[1].toFixed(2)})<br>
                            <small>Distance from origin: ${Math.sqrt(coord[0]*coord[0] + coord[1]*coord[1]).toFixed(2)}</small>
                        `;
                        showTooltip(event, tooltipText);
                    })
                    .on("mouseout", hideTooltip);
                
                // Add topic labels
                g.selectAll(".topic-label")
                    .data(coordinates)
                    .enter()
                    .append("text")
                    .attr("class", "topic-label")
                    .attr("x", d => xScale(d[0]))
                    .attr("y", d => yScale(d[1]))
                    .attr("text-anchor", "middle")
                    .attr("dy", "0.35em")
                    .style("font-size", "12px")
                    .style("font-weight", "bold")
                    .style("fill", "white")
                    .text((d, i) => i + 1)
                    .style("pointer-events", "none");
                
                // Create legend
                createTopicLegend();
            }
            
            function initTermChart() {
                const margin = {top: 30, right: 60, bottom: 50, left: 180};
                const width = 600 - margin.left - margin.right;
                const height = 500 - margin.top - margin.bottom;
                
                const svg = d3.select("#term-chart")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom);
                
                const g = svg.append("g")
                    .attr("transform", `translate(${margin.left},${margin.top})`);
                
                // Store references for updating
                window.termChartG = g;
                window.termChartDimensions = {width, height, margin};
            }
            
            function initControls() {
                const slider = d3.select("#lambda-slider");
                slider.on("input", function() {
                    currentLambda = +this.value;
                    d3.select("#lambda-value").text(currentLambda.toFixed(2));
                    updateTermChart();
                });
            }
            
            function selectTopic(topicIndex) {
                selectedTopic = topicIndex;
                
                // Update visual selection
                d3.selectAll(".topic-circle")
                    .classed("selected", (d, i) => i === topicIndex);
                
                updateTermChart();
            }
            
            function updateTermChart() {
                const g = window.termChartG;
                const {width, height} = window.termChartDimensions;
                
                // Clear existing bars
                g.selectAll("*").remove();
                
                // Get terms to display
                const terms = getTopTerms(selectedTopic, currentLambda);
                
                
                if (terms.length === 0) {
                    g.append("text")
                        .attr("x", width / 2)
                        .attr("y", height / 2)
                        .attr("text-anchor", "middle")
                        .style("font-size", "16px")
                        .style("fill", "#999")
                        .text("No terms to display");
                    return;
                }
                
                // Scales with better spacing
                const yScale = d3.scaleBand()
                    .domain(terms.map(d => d.Term))
                    .range([0, height])
                    .padding(0.2); // More spacing between bars
                
                const xScale = d3.scaleLinear()
                    .domain([0, d3.max(terms, d => Math.max(d.Total, d.Freq))])
                    .range([0, width]);
                
                // Add background bars (total frequency) with tooltips
                g.selectAll(".total-bar")
                    .data(terms)
                    .enter()
                    .append("rect")
                    .attr("class", "total-bar term-bar")
                    .attr("x", 0)
                    .attr("y", d => yScale(d.Term))
                    .attr("width", d => xScale(d.Total))
                    .attr("height", yScale.bandwidth())
                    .attr("fill", "#e2e8f0")
                    .attr("rx", 3)
                    .attr("ry", 3)
                    .on("mouseover", function(event, d) {
                        const tooltipText = `
                            <strong>"${d.Term}"</strong><br>
                            <span style="color: #9ca3af;">📊 Total Corpus Frequency:</span> ${d.Total.toLocaleString()}<br>
                            <small>How often this word appears across all documents</small>
                        `;
                        showTooltip(event, tooltipText);
                    })
                    .on("mouseout", hideTooltip);
                
                // Add topic-specific bars (if topic selected) with enhanced tooltips
                if (selectedTopic !== null) {
                    // Filter terms that have meaningful frequency values
                    const topicTerms = terms.filter(d => d.Freq && d.Freq > 0);
                    
                    if (topicTerms.length > 0) {
                        const topicBars = g.selectAll(".topic-bar")
                            .data(topicTerms, d => d.Term); // Use key function for proper data binding
                        
                        topicBars.enter()
                            .append("rect")
                            .attr("class", "topic-bar term-bar")
                            .attr("x", 0)
                            .attr("y", d => yScale(d.Term))
                            .attr("width", d => xScale(d.Freq || 0))
                            .attr("height", yScale.bandwidth())
                            .attr("fill", "#3b82f6")
                            .attr("rx", 3)
                            .attr("ry", 3)
                            .on("mouseover", function(event, d) {
                                const topicPercentage = ((d.Freq / d.Total) * 100).toFixed(1);
                                const relevanceScore = (currentLambda * d.logprob + (1 - currentLambda) * d.loglift).toFixed(3);
                                const isTopicSpecific = d.Freq / d.Total > 0.3; // More than 30% of occurrences in this topic
                                
                                const tooltipText = `
                                    <strong>"${d.Term}"</strong> in Topic ${selectedTopic + 1}<br>
                                    <span style="color: #3b82f6;">🔵 Topic Frequency:</span> ${d.Freq.toLocaleString()}<br>
                                    <span style="color: #9ca3af;">📊 Total Frequency:</span> ${d.Total.toLocaleString()}<br>
                                    <span style="color: #f59e0b;">📈 Topic Concentration:</span> ${topicPercentage}%<br>
                                    <span style="color: #10b981;">⭐ Relevance Score:</span> ${relevanceScore}<br>
                                    <small>${isTopicSpecific ? '🎯 Topic-specific term' : '🌐 General term'}</small>
                                `;
                                showTooltip(event, tooltipText);
                            })
                            .on("mouseout", hideTooltip);
                    }
                }
                
                // Add term labels with click-to-copy functionality
                g.selectAll(".term-label")
                    .data(terms)
                    .enter()
                    .append("text")
                    .attr("class", "term-label")
                    .attr("x", -5)
                    .attr("y", d => yScale(d.Term) + yScale.bandwidth() / 2)
                    .attr("text-anchor", "end")
                    .attr("dy", "0.35em")
                    .style("font-size", "12px")
                    .style("cursor", "pointer")
                    .text(d => d.Term)
                    .on("mouseover", function(event, d) {
                        if (selectedTopic !== null) {
                            const liftValue = Math.exp(d.loglift).toFixed(2);
                            const probValue = Math.exp(d.logprob).toFixed(4);
                            
                            const tooltipText = `
                                <strong>Click to copy: "${d.Term}"</strong><br>
                                <span style="color: #8b5cf6;">🔍 Probability in topic:</span> ${probValue}<br>
                                <span style="color: #f97316;">📊 Lift (vs corpus):</span> ${liftValue}x<br>
                                <small>Lift > 1 = more common in topic than overall</small>
                            `;
                            showTooltip(event, tooltipText);
                        } else {
                            const tooltipText = `
                                <strong>Click to copy: "${d.Term}"</strong><br>
                                <span style="color: #9ca3af;">📊 Total occurrences:</span> ${d.Total.toLocaleString()}<br>
                                <small>Select a topic to see detailed statistics</small>
                            `;
                            showTooltip(event, tooltipText);
                        }
                    })
                    .on("mouseout", hideTooltip)
                    .on("click", function(event, d) {
                        // Copy term to clipboard
                        navigator.clipboard.writeText(d.Term).then(() => {
                            // Show brief confirmation
                            const tooltipText = `<strong>✅ Copied "${d.Term}"</strong>`;
                            showTooltip(event, tooltipText);
                            setTimeout(() => hideTooltip(), 1000);
                        });
                    });
                
                // Add x-axis
                g.append("g")
                    .attr("class", "axis")
                    .attr("transform", `translate(0,${height})`)
                    .call(d3.axisBottom(xScale));
            }
            
            function getTopTerms(topicIndex, lambda) {
                const R = Math.min(30, visData.vocab.length);
                
                if (topicIndex === null) {
                    // Show overall most frequent terms
                    const defaultTerms = visData.topic_info
                        .filter(d => d.Category === 'Default')
                        .sort((a, b) => b.Total - a.Total)
                        .slice(0, R);
                    return defaultTerms;
                } else {
                    // Show terms for selected topic with relevance weighting
                    const topicCategory = `Topic${topicIndex + 1}`;
                    const topicTerms = visData.topic_info
                        .filter(d => d.Category === topicCategory && d.Freq > 1e-10) // Filter meaningful terms
                        .map(d => {
                            // Calculate relevance score (SAME FORMULA as calculate_term_relevance)
                            const relevance = lambda * d.logprob + (1 - lambda) * d.loglift;
                            return {...d, relevance};
                        })
                        .sort((a, b) => b.relevance - a.relevance)
                        .slice(0, R);
                    
                    return topicTerms;
                }
            }
            
            function createTopicLegend() {
                const legend = d3.select("#topic-legend");
                const legendItems = legend.selectAll(".legend-item")
                    .data(visData.topic_sizes)
                    .enter()
                    .append("div")
                    .attr("class", "legend-item");
                
                legendItems.append("span")
                    .attr("class", "legend-color")
                    .style("background-color", (d, i) => colorScale(i));
                
                legendItems.append("span")
                    .text((d, i) => `Topic ${i + 1} (${Math.round(d)} docs)`);
            }
            
            function showTooltip(event, html) {
                const tooltip = d3.select("#tooltip");
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(html)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }
            
            function hideTooltip() {
                d3.select("#tooltip").transition()
                    .duration(500)
                    .style("opacity", 0);
            }
        </script>
    </body>
    </html>
    """
    
    return html_template


def _create_vocab_from_tokenizer(tokenizer, n_vocab: int, emoji_map = None) -> List[str]:
    """
    Create vocabulary list from tokenizer, handling emoji decoding and filtering.
    
    Args:
        tokenizer: Turkish tokenizer object
        n_vocab: Size of vocabulary needed
        emoji_map: Emoji map for decoding (optional)
        
    Returns:
        List of vocabulary words
    """
    vocab = []
    
    for word_id in range(n_vocab):
        try:
            word = tokenizer.id_to_token(word_id)
            
            # Handle emoji decoding
            if emoji_map is not None and word is not None:
                if emoji_map.check_if_text_contains_tokenized_emoji(word):
                    word = emoji_map.decode_text(word)
            
            # Use the word as-is (don't filter out ## tokens for LDAvis display)
            # LDAvis is meant to show all tokens that the model uses
            if word is not None:
                vocab.append(word)
            else:
                vocab.append(f"[UNK_{word_id}]")  # Fallback for unknown tokens
                
        except Exception as e:
            # Fallback for any errors
            vocab.append(f"[ERROR_{word_id}]")
    
    return vocab


