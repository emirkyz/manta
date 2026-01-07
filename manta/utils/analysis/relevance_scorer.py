from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


def _get_word_cluster_for_doc_cluster(s_matrix: np.ndarray, doc_cluster_idx: int) -> int:
    """
    For a given doc-cluster (W column), find the best matching word-cluster (H row).

    S matrix structure: S[j, i] = coupling between W column i and H row j
    - Column i corresponds to doc-cluster i (W[:, i])
    - Row j corresponds to word-cluster j (H[j, :])

    Args:
        s_matrix: S matrix (k x k) where S[j, i] = coupling between W[:,i] and H[j,:]
        doc_cluster_idx: Index of the document cluster (W column)

    Returns:
        Index of the best matching word cluster (H row)
    """
    # Find word-cluster (row j) with maximum coupling to this doc-cluster (column i)
    return np.argmax(s_matrix[doc_cluster_idx, :])


def _create_vocab_from_tokenizer(tokenizer, n_vocab: int, emoji_map=None) -> List[str]:
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


def calculate_term_relevance(h_matrix: np.ndarray,
                             vocab: List[str] = None,
                             term_frequency: Optional[np.ndarray] = None,
                             w_matrix: Optional[np.ndarray] = None,
                             topic_idx: Optional[int] = None,
                             lambda_val: float = 0.6,
                             top_n: int = 30,
                             tokenizer=None,
                             emoji_map=None,
                             topic_word_idx: Optional[int] = None,
                             topic_doc_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate term relevance scores for a topic and return sorted terms.
    Replicates the JavaScript getTopTerms function logic from the LDAvis visualization.

    Args:
        h_matrix: Topic-word matrix (n_topics x n_vocab) or (k2 x n_vocab) for NMTF
        vocab: List of vocabulary words (optional if tokenizer provided)
        term_frequency: Array of term frequencies in corpus (optional)
        w_matrix: Document-topic matrix for calculating term frequency (optional)
        topic_idx: Index of topic to analyze (None for overall terms). Used for standard NMF.
        lambda_val: Lambda parameter for relevance calculation (0 to 1)
                   - 0: Most topic-specific (lift-based)
                   - 1: Most frequent in topic (probability-based)
                   - 0.6: Balanced (default)
        top_n: Number of top terms to return
        tokenizer: Tokenizer for creating vocabulary (optional)
        emoji_map: Emoji map for decoding (optional)
        topic_word_idx: Index of word topic in H matrix (for NMTF). Overrides topic_idx if provided.
        topic_doc_idx: Index of document topic in W matrix (for NMTF). Used with topic_word_idx.

    Returns:
        DataFrame with columns: Term, Freq, Total, logprob, loglift, relevance
        Sorted by relevance score in descending order
    """
    # Ensure matrices are dense numpy arrays
    if hasattr(h_matrix, 'toarray'):
        h_matrix = h_matrix.toarray()
    h_matrix = np.asarray(h_matrix)

    n_topics, n_vocab = h_matrix.shape

    # Create vocabulary from tokenizer if needed
    if tokenizer is not None:
        vocab = _create_vocab_from_tokenizer(tokenizer, n_vocab, emoji_map)

    if vocab is None or len(vocab) != n_vocab:
        raise ValueError(f"Vocabulary size mismatch: expected {n_vocab}, got {len(vocab) if vocab else 0}")

    # Calculate term frequencies if not provided
    if term_frequency is None:
        if w_matrix is not None:
            if hasattr(w_matrix, 'toarray'):
                w_matrix = w_matrix.toarray()
            w_matrix = np.asarray(w_matrix)

            # Weight by topic sizes for corpus representation
            topic_weights = w_matrix.sum(axis=0)
            term_frequency = np.sum(h_matrix * topic_weights.reshape(-1, 1), axis=0)
        else:
            # Fallback: use average across topics
            term_frequency = h_matrix.mean(axis=0)

    # Ensure term_frequency is numpy array
    term_frequency = np.asarray(term_frequency)

    # If no topic selected, return overall most frequent terms
    if topic_idx is None and topic_word_idx is None:
        df = pd.DataFrame({
            'Term': vocab,
            'Total': term_frequency,
            'Freq': term_frequency,  # Same as Total for overall view
            'Category': 'Default'
        })
        # Sort by total frequency and return top N
        df = df.sort_values('Total', ascending=False).head(top_n)
        df['relevance'] = df['Total']  # For consistency
        return df

    # Determine which topic index to use (NMTF vs standard NMF)
    # For NMTF: use topic_word_idx for H matrix, topic_doc_idx for W matrix
    # For standard NMF: use topic_idx for both
    if topic_word_idx is not None:
        # NMTF mode
        h_topic_idx = topic_word_idx
        w_topic_idx = topic_doc_idx if topic_doc_idx is not None else topic_word_idx
        display_topic_idx = topic_idx if topic_idx is not None else topic_word_idx
    else:
        # Standard NMF mode
        h_topic_idx = topic_idx
        w_topic_idx = topic_idx
        display_topic_idx = topic_idx

    # Validate topic index
    if h_topic_idx < 0 or h_topic_idx >= n_topics:
        raise ValueError(f"Invalid topic index: {h_topic_idx}. Must be between 0 and {n_topics - 1}")

    # Extract topic-specific data
    topic_word_vector = h_matrix[h_topic_idx]

    # Normalize to get probabilities
    topic_word_prob = topic_word_vector / (topic_word_vector.sum() + 1e-10)
    overall_word_prob = term_frequency / (term_frequency.sum() + 1e-10)

    # Calculate lift: ratio of word probability in topic to overall probability
    lift = topic_word_prob / (overall_word_prob + 1e-10)
    lift = np.clip(lift, 1e-10, None)  # Prevent negative or zero lift

    # Calculate log values
    logprob = np.log(topic_word_prob + 1e-10)
    loglift = np.log(lift)

    # Calculate relevance score (same formula as JavaScript)
    relevance = lambda_val * logprob + (1 - lambda_val) * loglift

    # Calculate topic-term frequencies (for 'Freq' field)
    if w_matrix is not None:
        topic_weight = w_matrix[:, w_topic_idx].sum()
        term_topic_freq = topic_word_vector * topic_weight
    else:
        term_topic_freq = topic_word_vector

    # Create DataFrame
    df = pd.DataFrame({
        'Term': vocab,
        'Freq': term_topic_freq,
        'Total': term_frequency,
        'logprob': logprob,
        'loglift': loglift,
        'relevance': relevance,
        'Category': f'Topic{display_topic_idx + 1}'
    })

    # Filter out terms with zero frequency in topic (optional but recommended)
    df = df[df['Freq'] > 1e-10]

    # Sort by relevance score (descending) and return top N
    df = df.sort_values('relevance', ascending=False).head(top_n)

    return df


def get_topic_top_terms(h_matrix: np.ndarray,
                        vocab: List[str] = None,
                        term_frequency: Optional[np.ndarray] = None,
                        w_matrix: Optional[np.ndarray] = None,
                        lambda_val: float = 0.6,
                        top_n: int = 10,
                        tokenizer=None,
                        emoji_map=None,
                        s_matrix: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Get top terms for all topics based on relevance score.

    For NMTF, uses sequential doc-cluster indices as topics. For each topic i (W column i),
    finds the best matching word-cluster (H row) via S matrix.

    Topic ordering is sequential: Topic i uses W column i. This ensures consistency
    with temporal visualization and other outputs.

    Args:
        h_matrix: Topic-word matrix (n_topics x n_vocab). For NMTF, this should be the original
                 H matrix (not transformed).
        vocab: List of vocabulary words
        term_frequency: Array of term frequencies
        w_matrix: Document-topic matrix (n_docs x n_topics)
        lambda_val: Lambda parameter for relevance
        top_n: Number of top terms per topic
        tokenizer: Tokenizer for vocabulary creation
        emoji_map: Emoji map for decoding
        s_matrix: S matrix for NMTF (k x k). S[j, i] = coupling between W[:,i] and H[j,:].
                 Used to find the best H row for each W column (topic).

    Returns:
        Dictionary with format:
        {
            "topic_01": {"word1": score1, "word2": score2, ...},
            "topic_02": {"word1": score1, "word2": score2, ...},
            ...
        }
    """
    if hasattr(h_matrix, 'toarray'):
        h_matrix = h_matrix.toarray()
    h_matrix = np.asarray(h_matrix)

    if hasattr(w_matrix, 'toarray'):
        w_matrix = w_matrix.toarray()
    if w_matrix is not None:
        w_matrix = np.asarray(w_matrix)

    topic_terms = {}

    if s_matrix is not None:
        # NMTF mode: use sequential doc-cluster indices as topics
        # Map each doc-cluster (W column) to its best word-cluster (H row) via S matrix
        # S[j, i] = coupling between W column i and H row j
        if hasattr(s_matrix, 'toarray'):
            s_matrix = s_matrix.toarray()
        s_matrix = np.asarray(s_matrix)

        n_topics = w_matrix.shape[1] if w_matrix is not None else s_matrix.shape[1]

        for topic_idx in range(n_topics):
            # Find best word-cluster (H row j) for this doc-cluster (W column i)
            word_cluster_idx = _get_word_cluster_for_doc_cluster(s_matrix, topic_idx)

            df = calculate_term_relevance(
                h_matrix=h_matrix,
                vocab=vocab,
                term_frequency=term_frequency,
                w_matrix=w_matrix,
                topic_idx=topic_idx,  # For display purposes
                topic_word_idx=word_cluster_idx,  # Use best H row for this topic
                topic_doc_idx=topic_idx,          # Use sequential W column
                lambda_val=lambda_val,
                top_n=top_n,
                tokenizer=tokenizer,
                emoji_map=emoji_map
            )

            # Format topic name with zero-padding (topic_01, topic_02, etc.)
            topic_name = f"topic_{topic_idx + 1:02d}"

            # Create dictionary of word:score pairs
            topic_terms[topic_name] = dict(zip(df['Term'], df['relevance'].round(4)))
    else:
        # Standard NMF mode: iterate sequentially
        n_topics = h_matrix.shape[0]

        for topic_idx in range(n_topics):
            df = calculate_term_relevance(
                h_matrix=h_matrix,
                vocab=vocab,
                term_frequency=term_frequency,
                w_matrix=w_matrix,
                topic_idx=topic_idx,
                lambda_val=lambda_val,
                top_n=top_n,
                tokenizer=tokenizer,
                emoji_map=emoji_map
            )

            # Format topic name with zero-padding (topic_01, topic_02, etc.)
            topic_name = f"topic_{topic_idx + 1:02d}"

            # Create dictionary of word:score pairs
            topic_terms[topic_name] = dict(zip(df['Term'], df['relevance'].round(4)))

    return topic_terms

