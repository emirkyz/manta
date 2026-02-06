"""
Gensim C_V Coherence Calculator

Clean, standalone module for calculating Gensim C_V coherence scores
directly from NMF/NMTF matrices.
"""

import multiprocessing as mp
from typing import Any, Dict, List, Optional

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def _fix_multiprocessing_fork():
    """Fix multiprocessing fork issue on macOS."""
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass


def _get_word_cluster_for_doc_cluster(s_matrix: np.ndarray, doc_cluster_idx: int) -> int:
    """
    For NMTF: find the best matching word-cluster (H row) for a doc-cluster (W column).

    S matrix structure: S[j, i] = coupling between W column i and H row j

    Args:
        s_matrix: S matrix (k x k)
        doc_cluster_idx: Index of the document cluster (W column)

    Returns:
        Index of the best matching word cluster (H row)
    """
    return np.argmax(s_matrix[doc_cluster_idx, :])


def _extract_topic_word_scores_with_relevance(
    h_matrix: np.ndarray,
    w_matrix: np.ndarray,
    vocabulary: List[str],
    s_matrix: Optional[np.ndarray],
    lambda_val: float,
    top_n: int
) -> Dict[str, Dict[str, float]]:
    """
    Extract top words for each topic using LDAvis-style relevance scoring.

    Relevance = lambda * log(P(word|topic)) + (1-lambda) * log(lift)
    where lift = P(word|topic) / P(word)

    Args:
        h_matrix: Topic-word matrix (n_topics x n_vocab)
        w_matrix: Document-topic matrix (n_docs x n_topics)
        vocabulary: List of vocabulary words
        s_matrix: Optional S matrix for NMTF
        lambda_val: Lambda for relevance (0-1)
        top_n: Number of top words per topic

    Returns:
        Dict mapping topic names to dict of word:relevance_score pairs
    """
    h_matrix = np.asarray(h_matrix)
    w_matrix = np.asarray(w_matrix)

    n_vocab = h_matrix.shape[1]

    # Calculate overall term frequency (weighted by topic sizes)
    topic_weights = w_matrix.sum(axis=0)
    term_frequency = np.sum(h_matrix * topic_weights.reshape(-1, 1), axis=0)
    overall_word_prob = term_frequency / (term_frequency.sum() + 1e-10)

    topic_word_scores = {}

    if s_matrix is not None:
        # NMTF mode
        s_matrix = np.asarray(s_matrix)
        n_topics = w_matrix.shape[1]

        for topic_idx in range(n_topics):
            word_cluster_idx = _get_word_cluster_for_doc_cluster(s_matrix, topic_idx)
            topic_word_vector = h_matrix[word_cluster_idx]

            words = _get_top_words_by_relevance(
                topic_word_vector, vocabulary, overall_word_prob, lambda_val, top_n
            )
            topic_word_scores[f"topic_{topic_idx + 1:02d}"] = words
    else:
        # Standard NMF mode
        n_topics = h_matrix.shape[0]

        for topic_idx in range(n_topics):
            topic_word_vector = h_matrix[topic_idx]
            #lambda_val = 0.5
            words = _get_top_words_by_relevance(
                topic_word_vector, vocabulary, overall_word_prob, lambda_val, top_n
            )
            topic_word_scores[f"topic_{topic_idx + 1:02d}"] = words

    return topic_word_scores


def _get_top_words_by_relevance(
    topic_word_vector: np.ndarray,
    vocabulary: List[str],
    overall_word_prob: np.ndarray,
    lambda_val: float,
    top_n: int
) -> Dict[str, float]:
    """
    Get top words for a single topic using relevance scoring.

    Args:
        topic_word_vector: Word scores for this topic
        vocabulary: List of vocabulary words
        overall_word_prob: Overall word probabilities in corpus
        lambda_val: Lambda for relevance calculation
        top_n: Number of top words to return

    Returns:
        Dict mapping words to their relevance scores, sorted by relevance
    """
    # Normalize to get topic-specific word probabilities
    topic_word_prob = topic_word_vector / (topic_word_vector.sum() + 1e-10)

    # Calculate lift and relevance
    lift = topic_word_prob / (overall_word_prob + 1e-10)
    lift = np.clip(lift, 1e-10, None)

    logprob = np.log(topic_word_prob + 1e-10)
    loglift = np.log(lift)

    relevance = lambda_val * logprob + (1 - lambda_val) * loglift

    # Get top N indices by relevance
    # Filter out words with zero frequency in topic
    valid_mask = topic_word_vector > 1e-10
    relevance_masked = np.where(valid_mask, relevance, -np.inf)

    top_indices = np.argsort(relevance_masked)[::-1][:top_n]

    # Extract words with scores, filtering subwords and handling "/" separator
    word_scores = {}
    for idx in top_indices:
        if relevance_masked[idx] > -np.inf and idx < len(vocabulary):
            word = vocabulary[idx]
            # Skip WordPiece subword tokens (## prefix)
            if word.startswith("##"):
                continue
            if "/" in word:
                word = word.split("/")[0].strip()
            if word:
                word_scores[word] = round(float(relevance[idx]), 4)

    return word_scores


def _tokenize_documents(documents: List[str]) -> List[List[str]]:
    """
    Tokenize documents by splitting on whitespace.

    Args:
        documents: List of document strings

    Returns:
        List of tokenized documents (list of word lists)
    """
    tokenized = []
    for doc in documents:
        if isinstance(doc, list):
            tokenized.append(doc)
        elif isinstance(doc, str):
            tokenized.append(doc.split())
        else:
            tokenized.append([])
    return tokenized


def calculate_gensim_cv_coherence(
    h_matrix: np.ndarray,
    w_matrix: np.ndarray,
    vocabulary: List[str],
    documents: List[str],
    s_matrix: Optional[np.ndarray] = None,
    lambda_val: float = 0.6,
    top_n_words: int = 15,
    processes: int = 1
) -> Dict[str, Any]:
    """
    Calculate Gensim C_V coherence score from NMF matrices.

    This function provides a clean interface for coherence calculation:
    1. Extracts topic words using LDAvis-style relevance scoring
    2. Tokenizes documents
    3. Computes C_V coherence using Gensim

    Args:
        h_matrix: Topic-word matrix (n_topics x n_vocab)
        w_matrix: Document-topic matrix (n_docs x n_topics)
        vocabulary: List of vocabulary words
        documents: List of document texts (strings or pre-tokenized lists)
        s_matrix: Optional S matrix for NMTF (k x k)
        lambda_val: Lambda for relevance scoring (0-1, default 0.6)
        top_n_words: Number of top words per topic for coherence (default 15)
        processes: Number of processes for Gensim (default 1)

    Returns:
        Dict with:
            - c_v_average: float - average coherence across topics
            - c_v_per_topic: Dict[str, float] - coherence per topic
            - topic_word_scores: Dict[str, Dict[str, float]] - word:relevance_score pairs per topic

    Raises:
        ValueError: If inputs are invalid
    """
    _fix_multiprocessing_fork()

    # Validate inputs
    if h_matrix is None or w_matrix is None:
        raise ValueError("h_matrix and w_matrix are required")
    if vocabulary is None or len(vocabulary) == 0:
        raise ValueError("vocabulary is required and cannot be empty")
    if documents is None or len(documents) == 0:
        raise ValueError("documents are required and cannot be empty")

    # Convert to numpy arrays
    h_matrix = np.asarray(h_matrix)
    w_matrix = np.asarray(w_matrix)
    if s_matrix is not None:
        s_matrix = np.asarray(s_matrix)

    # Extract topic words with relevance scores
    topic_word_scores = _extract_topic_word_scores_with_relevance(
        h_matrix=h_matrix,
        w_matrix=w_matrix,
        vocabulary=vocabulary,
        s_matrix=s_matrix,
        lambda_val=lambda_val,
        top_n=top_n_words
    )

    # Tokenize documents
    tokenized_docs = _tokenize_documents(documents)

    # Prepare topics list for Gensim (just the word lists, not scores)
    topics_list = [list(word_scores.keys()) for word_scores in topic_word_scores.values()]

    # Create Gensim dictionary from documents
    dictionary = Dictionary(tokenized_docs)

    # Calculate C_V coherence
    coherence_model = CoherenceModel(
        topics=topics_list,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
        topn=top_n_words,
        processes=processes
    )

    coherence_model_umass = CoherenceModel(
        topics=topics_list,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="u_mass",
        topn=top_n_words,
        processes=processes
    )


    c_v_average = coherence_model.get_coherence()
    c_v_per_topic_scores = coherence_model.get_coherence_per_topic()

    u_mass_average = coherence_model_umass.get_coherence()
    u_mass_per_topic_scores = coherence_model_umass.get_coherence_per_topic()

    # Build per-topic coherence dictionary
    topic_names = list(topic_word_scores.keys())
    c_v_per_topic = {}
    for i, score in enumerate(c_v_per_topic_scores):
        topic_name = topic_names[i] if i < len(topic_names) else f"topic_{i+1:02d}"
        c_v_per_topic[topic_name] = float(score) if not hasattr(score, 'tolist') else score.tolist()

    topic_names_umass = list(topic_word_scores.keys())
    u_mass_per_topic = {}
    for i, score in enumerate(u_mass_per_topic_scores):
        topic_name = topic_names_umass[i] if i < len(topic_names_umass) else f"topic_{i+1:02d}"
        u_mass_per_topic[topic_name] = float(score) if not hasattr(score, 'tolist') else score.tolist()


    return {
        "c_v_average": float(c_v_average),
        "c_v_per_topic": c_v_per_topic,
        "u_mass_average": float(u_mass_average),
        "u_mass_per_topic": u_mass_per_topic,
        "topic_word_scores": topic_word_scores,
    }
