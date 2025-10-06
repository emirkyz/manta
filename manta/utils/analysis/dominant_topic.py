import numpy as np


def get_dominant_topics(W, min_score=0.0):
    """
    Get the dominant topic for each document, filtering out zero-score documents.

    This function addresses the issue where np.argmax() assigns documents with all zero
    topic scores to topic 0 (Topic 1), which creates misleading visualizations and analysis.

    Args:
        W (numpy.ndarray): Document-topic matrix with shape (n_documents, n_topics).
                          Each row represents a document, each column represents a topic.
        min_score (float, optional): Minimum score threshold for a valid topic assignment.
                                    Documents with max score <= min_score are marked as -1.
                                    Default is 0.0.

    Returns:
        numpy.ndarray: Array of dominant topic indices with shape (n_documents,).
                      Values range from 0 to n_topics-1 for valid assignments,
                      or -1 for documents with no significant topic scores.

    Example:
        >>> W = np.array([[0.5, 0.3, 0.2],    # Doc 0 -> Topic 0
        ...               [0.0, 0.0, 0.0],    # Doc 1 -> -1 (no topic)
        ...               [0.1, 0.8, 0.1]])   # Doc 2 -> Topic 1
        >>> get_dominant_topics(W)
        array([ 0, -1,  1])

    Note:
        - Documents with all zero scores are assigned -1 (no dominant topic)
        - Visualizations should filter out documents with topic index -1
        - This prevents polluting topic distributions with meaningless assignments
    """
    # Convert to dense array if sparse
    if hasattr(W, 'toarray'):
        W = W.toarray()

    # Get the maximum score for each document
    max_scores = np.max(W, axis=1)

    # Get the dominant topic index (highest score)
    dominant_topics = np.argmax(W, axis=1)

    # Mark documents with zero or very low scores as -1 (no dominant topic)
    dominant_topics[max_scores <= min_score] = -1

    return dominant_topics
