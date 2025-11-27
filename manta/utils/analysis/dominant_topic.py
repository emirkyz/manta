import numpy as np


def _sort_matrices(s: np.ndarray) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Sort S matrix to find strongest topic-topic relationships for NMTF.

    For each column in S matrix, finds the row with maximum value.
    Returns topic pairs (word_topic_id, doc_topic_id) sorted by connection strength.

    Args:
        s (numpy.ndarray): S matrix from NMTF with shape (n_topics, n_topics)

    Returns:
        tuple: (ind, max_values) where ind is list of (word_topic_id, doc_topic_id) pairs
               and max_values is the corresponding connection strengths
    """
    ind = []
    max_values = []

    for i in range(s.shape[1]):
        col = s[:, i]
        max_ind = np.argmax(col)
        max_values.append(col[max_ind])
        ind.append((max_ind, i))

    ind_sorted = np.argsort(max_values)[::-1]
    ind = np.array(ind)[ind_sorted]
    max_values = np.array(max_values)[ind_sorted]

    return ind, max_values


def get_dominant_topics(W, min_score=0.0, s_matrix=None, H=None):
    """
    Get the dominant topic for each document, filtering out zero-score documents.
    Supports both standard NMF and NMTF models.

    This function addresses the issue where np.argmax() assigns documents with all zero
    topic scores to topic 0 (Topic 1), which creates misleading visualizations and analysis.

    For NMTF models (when s_matrix is provided), topics are defined by relationships in the
    S matrix. The function uses sorted topic pairs to properly assign documents to topics.
    The S matrix is expected to be L1 column-normalized (each column sums to 1.0) for
    consistent probability-like interpretation of topic relationships.

    Args:
        W (numpy.ndarray): Document-topic matrix with shape (n_documents, n_topics).
                          Each row represents a document, each column represents a topic.
        min_score (float, optional): Minimum score threshold for a valid topic assignment.
                                    Documents with max score <= min_score are marked as -1.
                                    Default is 0.0.
        s_matrix (numpy.ndarray, optional): S matrix for NMTF with shape (n_topics, n_topics).
                                           If provided, uses NMTF-specific topic extraction.
                                           Expected to be L1 column-normalized (columns sum to 1.0).
        H (numpy.ndarray, optional): H matrix for NMTF. Currently not used but kept for
                                    potential future enhancements.

    Returns:
        numpy.ndarray: Array of dominant topic indices with shape (n_documents,).
                      Values range from 0 to n_topics-1 for valid assignments,
                      or -1 for documents with no significant topic scores.

    Example (Standard NMF):
        >>> W = np.array([[0.5, 0.3, 0.2],    # Doc 0 -> Topic 0
        ...               [0.0, 0.0, 0.0],    # Doc 1 -> -1 (no topic)
        ...               [0.1, 0.8, 0.1]])   # Doc 2 -> Topic 1
        >>> get_dominant_topics(W)
        array([ 0, -1,  1])

    Example (NMTF with normalized S matrix):
        >>> W = np.array([[0.5, 0.3], [0.2, 0.8]])
        >>> S = np.array([[0.9, 0.1], [0.2, 0.7]])  # L1 column-normalized
        >>> dominant_topics = get_dominant_topics(W, s_matrix=S)
        # Uses normalized S matrix for consistent topic weighting

    Note:
        - Documents with all zero scores are assigned -1 (no dominant topic)
        - Visualizations should filter out documents with topic index -1
        - For NMTF, topic indices correspond to sorted S matrix pairs
        - S matrix should be column-normalized (L1 norm) for consistent interpretation
        - This prevents polluting topic distributions with meaningless assignments
    """
    # Convert to dense array if sparse
    if hasattr(W, 'toarray'):
        W = W.toarray()

    # NMTF mode: Use S matrix to determine topic assignments
    if s_matrix is not None:
        # Sort S matrix to get topic pairs by strength
        topic_pairs, pair_strengths = _sort_matrices(s_matrix)

        n_documents = W.shape[0]
        n_topics = len(topic_pairs)

        # Create topic score matrix for each document
        # Each column represents a "combined topic" defined by S matrix pairs
        topic_scores = np.zeros((n_documents, n_topics))

        for topic_idx, (word_topic_id, doc_topic_id) in enumerate(topic_pairs):
            # For each document, get score from W using doc_topic_id
            # Weight by the S matrix connection strength
            topic_scores[:, topic_idx] = W[:, doc_topic_id] * pair_strengths[topic_idx]

        # Get the maximum score for each document
        max_scores = np.max(topic_scores, axis=1)

        # Get the dominant topic index (highest score)
        dominant_topics = np.argmax(topic_scores, axis=1)

        # Mark documents with zero or very low scores as -1 (no dominant topic)
        dominant_topics[max_scores <= min_score] = -1

        return dominant_topics

    # Standard NMF mode
    else:
        # Get the maximum score for each document
        max_scores = np.max(W, axis=1)

        # Get the dominant topic index (highest score)
        dominant_topics = np.argmax(W, axis=1)

        # Mark documents with zero or very low scores as -1 (no dominant topic)
        dominant_topics[max_scores <= min_score] = -1

        return dominant_topics
