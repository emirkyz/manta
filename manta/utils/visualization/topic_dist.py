
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
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

def gen_topic_dist(W, output_dir, table_name, s_matrix=None):
    """Generate a bar plot of the document distribution across topics.
    From the W matrix, first we get biggest value per row. This is the topic that the document is most associated with.
    Then we count the number of documents for each topic.
    Bar plot values should sum up to the number of documents.

    For NMTF models (when s_matrix is provided), topics are defined by S matrix relationships.

    Args:
        W (numpy.ndarray): The matrix of topic distributions.
        output_dir (str): The directory to save the plot.
        table_name (str): The name of the table.
        s_matrix (numpy.ndarray, optional): S matrix for NMTF models.
    """
    print("Calculating document distribution across topics...")

    # For NMTF models, reorder W columns based on topic pairing from S matrix
    if s_matrix is not None:
        W_for_topics = _reorder_W_by_pairing(W, s_matrix)
    else:
        W_for_topics = W

    # Get dominant topics, filtering out zero-score documents
    dominant_topics = get_dominant_topics(W_for_topics, min_score=0.0)

    # Filter out documents with no dominant topic (marked as -1)
    valid_mask = dominant_topics != -1
    valid_dominant_topics = dominant_topics[valid_mask]
    excluded_count = np.sum(~valid_mask)

    if excluded_count > 0:
        print(f"Excluded {excluded_count} documents with all zero topic scores from visualization")

    # Count number of documents per topic (only valid assignments)
    if len(valid_dominant_topics) > 0:
        topic_counts = np.bincount(valid_dominant_topics)
    else:
        topic_counts = np.array([0])
    
    # Print the counts
    print("\nNumber of documents per topic:")
    for topic_idx, count in enumerate(topic_counts):
        print(f"Topic {topic_idx + 1}: {count} documents")

    start_index = 1
    end_index = len(topic_counts) + 1
    # Create and save bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(start_index, end_index), topic_counts)
    plt.xlabel('Topic Number')
    plt.ylabel('Number of Documents')
    plt.title('Number of Documents per Topic')
    plt.grid(False)
    plt.xticks(range(start_index, end_index))
    
    # Add count labels on top of each bar
    for i, count in enumerate(topic_counts):
        plt.text(i+start_index, count, str(count), ha='center', va='bottom')
    
    # Check if output_dir already includes the table_name to avoid double nesting
    output_dir_path = Path(output_dir)
    if output_dir_path.name == table_name:
        table_output_dir = output_dir_path
    else:
        # Create table-specific subdirectory under output folder
        table_output_dir = output_dir_path / table_name
    table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot to table-specific subdirectory
    plot_path = table_output_dir / f"{table_name}_document_dist.png"
    plt.savefig(plot_path,dpi=1000)
    print(f"Document distribution plot saved to: {plot_path}")
    return plt, topic_counts