from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


def _sort_matrices(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort NMTF S matrix columns by their maximum connection strength.
    Returns topic pairs (word_topic_id, doc_topic_id) sorted by connection strength.

    This replicates the logic from topic_extractor.py to ensure consistent topic ordering.

    Args:
        s: S matrix from NMTF with shape (k1, k2) where k1=doc topics, k2=word topics

    Returns:
        Tuple of (sorted_pairs, max_values) where:
            - sorted_pairs: array of (word_topic_idx, doc_topic_idx) tuples
            - max_values: array of maximum connection strengths
    """
    ind = []
    max_values = []

    for i in range(s.shape[1]):
        col = s[:, i]
        max_ind = np.argmax(col)
        max_values.append(col[max_ind])
        ind.append((i, max_ind))

    ind_sorted = np.argsort(max_values)[::-1]
    ind = np.array(ind)[ind_sorted]
    max_values = np.array(max_values)[ind_sorted]

    return ind, max_values


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

    Args:
        h_matrix: Topic-word matrix (n_topics x n_vocab) or (k2 x n_vocab) for NMTF
        vocab: List of vocabulary words
        term_frequency: Array of term frequencies
        w_matrix: Document-topic matrix (n_docs x n_topics) or (n_docs x k1) for NMTF
        lambda_val: Lambda parameter for relevance
        top_n: Number of top terms per topic
        tokenizer: Tokenizer for vocabulary creation
        emoji_map: Emoji map for decoding
        s_matrix: S matrix for NMTF (k1 x k2). If provided, uses NMTF-aware topic extraction

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

    # NMTF mode: use S matrix to determine topic ordering and connections
    if s_matrix is not None:
        if hasattr(s_matrix, 'toarray'):
            s_matrix = s_matrix.toarray()
        s_matrix = np.asarray(s_matrix)

        # Sort S matrix to get topic pairs (word_vec_id, doc_vec_id) by connection strength
        topic_pairs, max_vals = _sort_matrices(s_matrix)

        # Extract topics based on S matrix connections
        for idx, (word_vec_id, doc_vec_id) in enumerate(topic_pairs):
            df = calculate_term_relevance(
                h_matrix=h_matrix,
                vocab=vocab,
                term_frequency=term_frequency,
                w_matrix=w_matrix,
                topic_idx=idx,  # Display index (0, 1, 2, ...)
                lambda_val=lambda_val,
                top_n=top_n,
                tokenizer=tokenizer,
                emoji_map=emoji_map,
                topic_word_idx=word_vec_id,  # Actual H matrix row to use
                topic_doc_idx=doc_vec_id  # Actual W matrix column to use
            )

            # Format topic name with zero-padding (topic_01, topic_02, etc.)
            topic_name = f"topic_{idx + 1:02d}"

            # Create dictionary of word:score pairs
            topic_terms[topic_name] = dict(zip(df['Term'], df['relevance'].round(4)))

    # Standard NMF mode: iterate through all topics sequentially
    else:
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

