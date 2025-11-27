"""
Simplified coherence scoring for NMF topic models.

This module provides coherence metrics to evaluate topic quality:
- c_v: Sliding window-based coherence (via Gensim) - RECOMMENDED
- u_mass: Document-level co-occurrence coherence
- c_uci: UCI coherence using pointwise mutual information (PMI)

Usage:
    from manta.utils.analysis.nmf_coherence import calculate_nmf_coherence

    # Calculate c_v coherence (recommended)
    results = calculate_nmf_coherence(
        W=doc_topic_matrix,
        H=topic_word_matrix,
        texts=tokenized_documents,
        vocabulary=vocab_list,
        metrics=['c_v'],
        top_n=10
    )
    print(f"C_v coherence: {results['c_v']['average']:.4f}")
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from itertools import combinations

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from numpy import floating
from scipy.sparse import csc_matrix


# ============================================================================
# SECTION 1: HELPER FUNCTIONS
# ============================================================================

def prepare_gensim_data(topics_json: Dict, documents: List[List[str]]) -> Tuple[List[List[str]], Dictionary, List]:
    """
    Prepare data for Gensim's CoherenceModel.

    Args:
        topics_json: Dictionary mapping topic names to word-score dictionaries
        documents: List of tokenized documents (each document is a list of tokens)

    Returns:
        tuple: (topics_list, dictionary, corpus)
            - topics_list: List of top words for each topic
            - dictionary: Gensim Dictionary object
            - corpus: Bag-of-words corpus
    """
    # Prepare topics list
    topics_list = []
    for topic_id, word_scores in topics_json.items():
        # Handle words like "word1 / word2" by taking the first part
        top_words = []
        for word, score in word_scores.items():
            if "/" in word:
                words = word.split("/")
                top_words.append(words[0].strip())
            else:
                top_words.append(word)
        topics_list.append(top_words)

    # Ensure documents are properly tokenized
    if not documents or len(documents) == 0:
        raise ValueError("No documents provided")

    # Create dictionary and corpus
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    return topics_list, dictionary, corpus


def p_word(word: str, documents: List[List[str]]) -> float:
    """
    Calculate the probability of a word appearing in documents.
    P(w) = D(w) / N

    Args:
        word: The word to calculate probability for
        documents: List of tokenized documents

    Returns:
        float: Probability of word appearing in documents
    """
    D_w = sum(1 for doc in documents if word in doc)
    N = len(documents)
    return D_w / N if N > 0 else 0.0


def p_word_pair(word1: str, word2: str, documents: List[List[str]]) -> float:
    """
    Calculate the probability of a word pair co-occurring in documents.
    P(w1, w2) = D(w1, w2) / N

    Args:
        word1: First word
        word2: Second word
        documents: List of tokenized documents

    Returns:
        float: Probability of word pair co-occurring
    """
    D_w1_w2 = sum(1 for doc in documents if word1 in doc and word2 in doc)
    N = len(documents)
    return D_w1_w2 / N if N > 0 else 0.0


def pmi(word1: str, word2: str, documents: List[List[str]], epsilon: float = 1e-9) -> Union[float, str]:
    """
    Calculate pointwise mutual information (PMI) between two words.
    PMI(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))

    Args:
        word1: First word
        word2: Second word
        documents: List of tokenized documents
        epsilon: Small value to prevent log(0)

    Returns:
        float: PMI score, or "zero_division_error" if calculation fails
    """
    p1 = p_word(word1, documents)
    p2 = p_word(word2, documents)

    if p1 == 0 or p2 == 0:
        return "zero_division_error"

    p_pair = p_word_pair(word1, word2, documents)
    return math.log((p_pair + epsilon) / (p1 * p2))


# ============================================================================
# SECTION 2: U_MASS COHERENCE CLASS
# ============================================================================

class UMassCoherence:
    """
    Calculate U_mass coherence for topic models.

    U_mass is based on document co-occurrence and doesn't require external corpora.
    It uses a sliding window approach and logarithmic conditional probability.

    Formula: U_mass(w_i, w_j) = log((D(w_i, w_j) + epsilon) / D(w_j))
    where D(w_i, w_j) is the number of documents containing both words.
    """

    def __init__(self, documents: List[List[str]], epsilon: float = 1e-12):
        """
        Initialize U_mass coherence calculator.

        Args:
            documents: List of documents, where each document is a list of words
            epsilon: Small value to avoid log(0)
        """
        self.documents = documents
        self.epsilon = epsilon
        self.word_doc_freq = defaultdict(set)
        self.cooccur_freq = defaultdict(lambda: defaultdict(int))

        # Build word-document frequency and co-occurrence matrices
        self._build_frequencies()

    def _build_frequencies(self):
        """Build word-document frequency and co-occurrence frequency dictionaries."""
        for doc_id, doc in enumerate(self.documents):
            # Get unique words in document
            unique_words = set(doc)

            # Track which documents contain each word
            for word in unique_words:
                self.word_doc_freq[word].add(doc_id)

            # Track co-occurrences
            unique_words_list = list(unique_words)
            for i in range(len(unique_words_list)):
                for j in range(i + 1, len(unique_words_list)):
                    word1, word2 = unique_words_list[i], unique_words_list[j]
                    # Store co-occurrences symmetrically
                    self.cooccur_freq[word1][word2] += 1
                    self.cooccur_freq[word2][word1] += 1

    def calculate_umass_coherence(self, topic_words: Union[Dict, List], top_n: int = 10) -> float:
        """
        Calculate U_mass coherence for a topic.

        Args:
            topic_words: List of (word, score) tuples or dict of word scores
            top_n: Number of top words to consider

        Returns:
            float: U_mass coherence score
        """
        # Get top N words
        if isinstance(topic_words, dict):
            # Sort by score and get top N
            sorted_words = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)
            top_words = []
            for word, score in sorted_words[:top_n]:
                # Handle words with "/" separator (take first part)
                if "/" in word:
                    words = word.split("/")
                    top_words.append(words[0].strip())
                else:
                    top_words.append(word)
        else:
            # Assume it's already a list of words
            top_words = list(topic_words)[:top_n]

        coherence_score = 0.0
        pair_count = 0

        # Calculate pairwise coherence
        for i in range(1, len(top_words)):
            for j in range(i):
                word_i = top_words[i]
                word_j = top_words[j]

                # Get document frequencies
                D_wi = len(self.word_doc_freq.get(word_i, set()))
                D_wj = len(self.word_doc_freq.get(word_j, set()))

                # Get co-occurrence frequency
                D_wi_wj = self.cooccur_freq.get(word_i, {}).get(word_j, 0)

                # Calculate U_mass score for this pair
                if D_wi > 0 and D_wj > 0 and D_wi_wj > 0:
                    # U_mass formula: log((D(wi, wj) + epsilon) / D(wj))
                    score = math.log((D_wi_wj + self.epsilon) / D_wj)
                    coherence_score += score
                    pair_count += 1

        # Return average coherence
        if pair_count > 0:
            return coherence_score / pair_count
        else:
            return 0.0

    def calculate_all_topics_coherence(self, topics_dict: Dict, top_n: int = 10) -> Dict:
        """
        Calculate U_mass coherence for all topics.

        Args:
            topics_dict: Dictionary of topics with word scores
            top_n: Number of top words to consider per topic

        Returns:
            dict: Dictionary containing topic coherences and average coherence
        """
        topic_coherences = {}
        coherence_values = []

        for topic_name, topic_words in topics_dict.items():
            coherence_score = self.calculate_umass_coherence(topic_words, top_n)
            topic_coherences[f"{topic_name}_coherence"] = coherence_score
            coherence_values.append(coherence_score)

        average_coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0

        return {
            "topic_coherences": topic_coherences,
            "average_coherence": average_coherence
        }


# ============================================================================
# SECTION 3: COHERENCE METRIC FUNCTIONS
# ============================================================================

def calculate_cv_coherence(
        W: np.ndarray,
        H: np.ndarray,
        texts: List[List[str]],
        vocabulary: Optional[List[str]] = None,
        dictionary: Optional[Dictionary] = None,
        top_n: int = 10
) -> Dict:
    """
    Calculate c_v coherence using Gensim (RECOMMENDED metric).

    C_v coherence is based on a sliding window, one-set segmentation,
    and indirect confirmation measure. It has been shown to have the
    highest correlation with human judgment.

    Args:
        W: Document-topic matrix from NMF (n_docs, n_topics)
        H: Topic-word matrix from NMF (n_topics, n_words)
        texts: Original tokenized documents (list of token lists)
        vocabulary: List of words corresponding to H matrix columns
        dictionary: Gensim Dictionary object (optional, will be created if not provided)
        top_n: Number of top words per topic to evaluate

    Returns:
        dict: Contains 'average' score and 'per_topic' scores
    """
    # Get top words for each topic from H matrix
    topics = []
    n_topics = H.shape[0]

    for topic_idx in range(n_topics):
        # Get indices of top N words for this topic
        top_word_idx = H[topic_idx, :].argsort()[-top_n:][::-1]

        # Convert indices to words
        if vocabulary is not None:
            top_words = [vocabulary[idx] for idx in top_word_idx if idx < len(vocabulary)]
        elif dictionary is not None:
            top_words = [dictionary[idx] for idx in top_word_idx]
        else:
            raise ValueError("Either vocabulary or dictionary must be provided")

        topics.append(top_words)

    # Create dictionary if not provided
    if dictionary is None:
        dictionary = Dictionary(texts)

    # Calculate coherence
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    average_coherence = coherence_model.get_coherence()
    per_topic_coherence = coherence_model.get_coherence_per_topic()

    # Format results
    topic_coherence_dict = {}
    for i, score in enumerate(per_topic_coherence):
        topic_coherence_dict[f"topic_{i}"] = float(score) if hasattr(score, 'item') else score

    return {
        "average": average_coherence,
        "per_topic": topic_coherence_dict
    }


def calculate_umass_coherence(
        topics_dict: Dict,
        documents: List[List[str]],
        top_n: int = 10,
        epsilon: float = 1e-12
) -> Dict:
    """
    Calculate u_mass coherence using document co-occurrence.

    U_mass measures coherence based on document co-occurrence counts.
    It doesn't require external corpora and is faster to compute than c_v.

    Args:
        topics_dict: Dictionary mapping topic names to word-score dictionaries
        documents: List of tokenized documents (list of token lists)
        top_n: Number of top words per topic to evaluate
        epsilon: Smoothing parameter to avoid log(0)

    Returns:
        dict: Contains 'average' score and 'topic_coherences' dict
    """
    # Initialize UMassCoherence calculator
    umass_calc = UMassCoherence(documents, epsilon=epsilon)

    # Calculate coherence scores for all topics
    results = umass_calc.calculate_all_topics_coherence(topics_dict, top_n=top_n)

    return {
        "average": results["average_coherence"],
        "topic_coherences": results["topic_coherences"]
    }


def calculate_uci_coherence(
        topics_dict: Dict,
        documents: List[List[str]],
        top_n: int = 10,
        epsilon: float = 1e-9
) -> Dict:
    """
    Calculate c_uci coherence using pointwise mutual information (PMI).

    UCI coherence uses PMI to measure word association strength.
    Formula: UCI(topic) = (2 / (N * (N-1))) * sum_i sum_j PMI(w_i, w_j)

    Args:
        topics_dict: Dictionary mapping topic names to word-score dictionaries
        documents: List of tokenized documents (list of token lists)
        top_n: Number of top words per topic to evaluate
        epsilon: Smoothing parameter to prevent log(0)

    Returns:
        dict: Contains 'average' score and 'topic_coherences' dict
    """
    topic_coherences = {}
    total_coherence_sum = 0
    valid_topics_count = 0

    for topic_id, word_scores in topics_dict.items():
        # Sort words by their scores in descending order and take top words
        sorted_words = sorted(word_scores.items(), key=lambda x: float(x[1]), reverse=True)
        top_words = [word for word, _ in sorted_words[:top_n]]

        # Handle words with "/" separator
        processed_words = []
        for word in top_words:
            if "/" in word:
                processed_words.append(word.split("/")[0].strip())
            else:
                processed_words.append(word)

        N = len(processed_words)
        if N < 2:
            continue

        # Calculate PMI for all word pairs
        word_combinations = combinations(processed_words, 2)
        pmi_values = []

        for word1, word2 in word_combinations:
            pmi_val = pmi(word1, word2, documents, epsilon)
            if pmi_val != "zero_division_error":
                pmi_values.append(pmi_val)

        if pmi_values:
            # Calculate UCI coherence for this topic
            topic_coherence = sum(pmi_values) / len(pmi_values)
            topic_coherences[f"{topic_id}_coherence"] = topic_coherence
            total_coherence_sum += topic_coherence
            valid_topics_count += 1

    average_coherence = total_coherence_sum / valid_topics_count if valid_topics_count > 0 else 0.0

    return {
        "average": average_coherence,
        "topic_coherences": topic_coherences
    }


# ============================================================================
# SECTION 4: UNIFIED INTERFACE
# ============================================================================

def extract_topics_from_matrix(H: np.ndarray, vocabulary: List[str], top_n: int = 10) -> Dict:
    """
    Extract topic-word dictionaries from H matrix.

    Args:
        H: Topic-word matrix from NMF (n_topics, n_words)
        vocabulary: List of words corresponding to matrix columns
        top_n: Number of top words to extract per topic

    Returns:
        dict: Dictionary mapping topic_ids to word-score dictionaries
    """
    topics_dict = {}
    n_topics = H.shape[0]

    for topic_idx in range(n_topics):
        # Get indices of top N words
        top_word_idx = H[topic_idx, :].argsort()[-top_n:][::-1]

        # Create word-score dictionary
        topic_words = {}
        for idx in top_word_idx:
            if idx < len(vocabulary):
                word = vocabulary[idx]
                score = float(H[topic_idx, idx])
                if score > 0 and word:
                    topic_words[word] = score

        topics_dict[f"topic_{topic_idx}"] = topic_words

    return topics_dict


def calculate_nmf_coherence(
        W: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        texts: Optional[List[List[str]]] = None,
        vocabulary: Optional[List[str]] = None,
        topics_dict: Optional[Dict] = None,
        metrics: List[str] = ["c_v"],
        top_n: int = 10,
        epsilon: float = 1e-12
) -> Dict:
    """
    Calculate coherence scores for NMF topics (MAIN INTERFACE).

    This is the primary function for calculating coherence metrics.
    You can provide either NMF matrices (W, H) or a pre-computed topics dictionary.

    Args:
        W: Document-topic matrix from NMF (n_docs, n_topics) - optional
        H: Topic-word matrix from NMF (n_topics, n_words) - optional
        texts: Tokenized documents (list of token lists) - REQUIRED
        vocabulary: List of words corresponding to H matrix columns - required if H is provided
        topics_dict: Pre-computed topic-word dictionary - optional alternative to H
        metrics: List of metrics to calculate. Options: ["c_v", "u_mass", "c_uci"]
        top_n: Number of top words per topic to evaluate
        epsilon: Smoothing parameter for u_mass and c_uci

    Returns:
        dict: Dictionary containing results for each requested metric

    Example:
        >>> results = calculate_nmf_coherence(
        ...     W=doc_topic_matrix,
        ...     H=topic_word_matrix,
        ...     texts=tokenized_docs,
        ...     vocabulary=vocab_list,
        ...     metrics=['c_v', 'u_mass'],
        ...     top_n=10
        ... )
        >>> print(f"C_v: {results['c_v']['average']:.4f}")
        >>> print(f"U_mass: {results['u_mass']['average']:.4f}")
    """
    # Validate inputs
    if texts is None or len(texts) == 0:
        raise ValueError("texts (tokenized documents) are required")

    # Ensure texts are properly tokenized
    if isinstance(texts[0], str):
        texts = [doc.split() for doc in texts]

    # Extract topics from H matrix if provided
    if topics_dict is None:
        if H is None or vocabulary is None:
            raise ValueError("Either topics_dict or (H + vocabulary) must be provided")
        topics_dict = extract_topics_from_matrix(H, vocabulary, top_n)

    # Initialize results
    results = {}

    # Calculate requested metrics
    for metric in metrics:
        metric = metric.lower()

        if metric == "c_v":
            if H is None:
                raise ValueError("c_v coherence requires H matrix and W matrix")
            if W is None:
                raise ValueError("c_v coherence requires W matrix")

            results["c_v"] = calculate_cv_coherence(
                W=W,
                H=H,
                texts=texts,
                vocabulary=vocabulary,
                top_n=top_n
            )

        elif metric == "u_mass":
            results["u_mass"] = calculate_umass_coherence(
                topics_dict=topics_dict,
                documents=texts,
                top_n=top_n,
                epsilon=epsilon
            )

        elif metric == "c_uci":
            results["c_uci"] = calculate_uci_coherence(
                topics_dict=topics_dict,
                documents=texts,
                top_n=top_n,
                epsilon=epsilon
            )

        else:
            print(f"Warning: Unknown metric '{metric}'. Supported: c_v, u_mass, c_uci")

    return results


# ============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ============================================================================

# Alias for backwards compatibility with existing code
def u_mass(topics_json: Dict, documents: List[List[str]], top_n: int = 10, epsilon: float = 1e-12) -> Dict:
    """
    Backwards compatibility wrapper for calculate_umass_coherence.

    DEPRECATED: Use calculate_umass_coherence() or calculate_nmf_coherence() instead.
    """
    return calculate_umass_coherence(topics_json, documents, top_n, epsilon)


def c_uci(topics_json: Dict, documents: List[List[str]], top_n: int = 10, epsilon: float = 1e-9) -> Dict:
    """
    Backwards compatibility wrapper for calculate_uci_coherence.

    DEPRECATED: Use calculate_uci_coherence() or calculate_nmf_coherence() instead.
    """
    return calculate_uci_coherence(topics_json, documents, top_n, epsilon)


def calculate_reconstruct_error(
        X : csc_matrix,
        W: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None, ) -> floating[Any]:

    X_reconstructed = W @ H
    frobenius_norm = np.linalg.norm(X - X_reconstructed, 'fro')
    return frobenius_norm