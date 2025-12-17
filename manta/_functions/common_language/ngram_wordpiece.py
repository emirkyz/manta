"""
WordPiece-based N-gram implementation for word-level merging.

This module implements the WordPiece algorithm that operates on counterized (numerical)
word data. Unlike BPE which uses frequency counts, WordPiece uses likelihood scores
to select word pairs for merging, maximizing the language model likelihood.

The likelihood score for a pair (word1, word2) is calculated as:
    score = freq(word1, word2) / (freq(word1) * freq(word2))

This score measures how much more likely the pair occurs together compared to
if the words were independent.

Example:
    If "good" and "product" occur together 100 times, "good" occurs 200 times,
    and "product" occurs 150 times, the likelihood score would be:
    100 / (200 * 150) = 0.00333

    Higher scores indicate stronger associations.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import json
import os
import time
from tqdm import tqdm
import heapq

if TYPE_CHECKING:
    from ...utils.console.console_manager import ConsoleManager

# Try to import numba for JIT compilation (optional dependency)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# JIT-compiled helper functions for maximum performance
@jit(nopython=True, cache=True)
def count_pairs_numba(doc_array):
    """
    Numba-optimized pair counting for a single document.

    Args:
        doc_array: numpy array of word IDs

    Returns:
        2D array where each row is [word1, word2]
    """
    if len(doc_array) < 2:
        return np.empty((0, 2), dtype=doc_array.dtype)

    # Create pairs array efficiently
    n_pairs = len(doc_array) - 1
    pairs = np.empty((n_pairs, 2), dtype=doc_array.dtype)

    for i in range(n_pairs):
        pairs[i, 0] = doc_array[i]
        pairs[i, 1] = doc_array[i + 1]

    return pairs


@jit(nopython=True, cache=True)
def count_pairs_from_docs_numba(docs_list):
    """
    Count all pairs across multiple documents efficiently.

    Args:
        docs_list: List of numpy arrays (documents)

    Returns:
        2D array of all pairs
    """
    # First pass: count total pairs
    total_pairs = 0
    for doc in docs_list:
        if len(doc) >= 2:
            total_pairs += len(doc) - 1

    if total_pairs == 0:
        return np.empty((0, 2), dtype=docs_list[0].dtype)

    # Second pass: collect pairs
    all_pairs = np.empty((total_pairs, 2), dtype=docs_list[0].dtype)
    idx = 0

    for doc in docs_list:
        if len(doc) >= 2:
            for i in range(len(doc) - 1):
                all_pairs[idx, 0] = doc[i]
                all_pairs[idx, 1] = doc[i + 1]
                idx += 1

    return all_pairs


@jit(nopython=True, cache=True)
def count_words_numba(doc_array):
    """
    Numba-optimized word counting for a single document.

    Args:
        doc_array: numpy array of word IDs

    Returns:
        Array of unique word IDs in the document
    """
    # Return all words (not unique, for frequency counting)
    return doc_array


class WordPieceNGram:
    """
    WordPiece algorithm for creating n-gram word pairs from counterized data.

    This class implements a likelihood-based approach to generating meaningful
    word n-grams by iteratively merging pairs with the highest likelihood scores
    until a vocabulary size limit is reached.
    """

    def __init__(self, vocab_limit: int = 10000, min_likelihood_score: float = 0.0,
                 smoothing: float = 1e-10, verbose: bool = False,
                 console: Optional["ConsoleManager"] = None):
        """
        Initialize the WordPiece N-gram encoder.

        Args:
            vocab_limit: Maximum vocabulary size before stopping merging
            min_likelihood_score: Minimum likelihood threshold for pair merging
            smoothing: Small constant to prevent division by zero (default: 1e-10)
            verbose: Whether to print detailed decoding information (default: False for speed)
            console: Console manager for output. If None, uses global console.
        """
        # Import here to avoid circular imports
        from ...utils.console.console_manager import get_console
        self._console = console or get_console()

        self.vocab_limit = vocab_limit
        self.min_likelihood_score = min_likelihood_score
        self.smoothing = smoothing
        self.verbose = verbose
        self.original_vocab_size = 0
        self.current_vocab_size = 0
        self.merge_operations = []  # List of (pair, new_id, likelihood_score, frequency) tuples
        self.pair_to_id = {}  # Mapping of merged pairs to their new IDs
        self.id_to_pair = {}  # Reverse mapping for reconstruction
        self.word_frequencies = {}  # Track individual word frequencies
        self.pair_frequencies = {}  # Track pair frequencies
        self.likelihood_scores = {}  # Cache computed likelihood scores
        self.inverted_index = {}  # Mapping of word_id -> set of doc indices containing that word

        # Optimization: Track max likelihood pair for O(1) lookup
        self.max_likelihood = -float('inf')
        self.max_pair = None

    def _get_optimal_dtype(self):
        """
        Determine optimal numpy dtype based on vocabulary size.

        Returns:
            numpy dtype (int16 or int32) based on vocab_limit
        """
        # int16 max value is 32767, int32 max is 2147483647
        if self.vocab_limit <= 32767:
            return np.int16
        else:
            return np.int32

    def build_inverted_index(self, counterized_data: List[List[int]]) -> Dict[int, set]:
        """
        Build inverted index mapping word IDs to document indices.

        Args:
            counterized_data: List of documents with word IDs

        Returns:
            Dictionary mapping word_id -> set of document indices
        """
        inverted_index = defaultdict(set)

        for doc_idx, document in enumerate(counterized_data):
            # Add each unique word in document to the inverted index
            for word_id in set(document):
                inverted_index[word_id].add(doc_idx)

        return dict(inverted_index)

    def calculate_word_frequencies(self, counterized_data: List[List[int]]) -> Dict[int, int]:
        """
        Calculate frequency of each individual word across all documents.

        Args:
            counterized_data: List of documents with word IDs

        Returns:
            Dictionary mapping word_id -> frequency
        """
        word_freq = {}

        if NUMBA_AVAILABLE:
            # Use Numba-optimized counting
            for doc_array in counterized_data:
                words = count_words_numba(doc_array)
                for word_id in words:
                    word_freq[word_id] = word_freq.get(word_id, 0) + 1
        else:
            # Fallback to standard counting
            for document in counterized_data:
                for word_id in document:
                    word_freq[word_id] = word_freq.get(word_id, 0) + 1

        return word_freq

    def build_pair_frequency_table(self, counterized_data: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """
        Build frequency table of adjacent word pairs.

        Args:
            counterized_data: List of documents with word IDs

        Returns:
            Dictionary with (word1_id, word2_id) -> frequency mappings
        """
        if NUMBA_AVAILABLE:
            # Use Numba-optimized batch pair extraction
            all_pairs = count_pairs_from_docs_numba(counterized_data)

            # Convert pairs array to tuples and count frequencies
            pair_frequencies = {}
            for i in range(len(all_pairs)):
                pair = (all_pairs[i, 0], all_pairs[i, 1])
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1

            return pair_frequencies
        else:
            # Fallback to vectorized numpy
            pair_frequencies = {}

            for doc_array in counterized_data:
                if len(doc_array) < 2:
                    continue

                # Use numpy slicing
                pairs = np.column_stack([doc_array[:-1], doc_array[1:]])
                for i in range(len(pairs)):
                    pair = (pairs[i, 0], pairs[i, 1])
                    pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1

            return pair_frequencies

    def calculate_likelihood_score(self, pair: Tuple[int, int],
                                   pair_freq: int, word1_freq: int, word2_freq: int) -> float:
        """
        Calculate WordPiece likelihood score for a pair.

        Formula: score = freq(pair) / (freq(word1) * freq(word2))

        Args:
            pair: The (word1_id, word2_id) tuple
            pair_freq: Frequency of the pair occurring together
            word1_freq: Frequency of word1 alone
            word2_freq: Frequency of word2 alone

        Returns:
            Likelihood score (higher is better)
        """
        # Prevent division by zero with smoothing
        denominator = (word1_freq + self.smoothing) * (word2_freq + self.smoothing)
        return pair_freq / denominator

    def calculate_all_likelihood_scores(self, pair_frequencies: Dict[Tuple[int, int], int],
                                       word_frequencies: Dict[int, int]) -> Dict[Tuple[int, int], float]:
        """
        Calculate likelihood scores for all pairs.

        Args:
            pair_frequencies: Dictionary of pair frequencies
            word_frequencies: Dictionary of word frequencies

        Returns:
            Dictionary mapping pairs to their likelihood scores
        """
        likelihood_scores = {}

        for pair, pair_freq in pair_frequencies.items():
            word1_id, word2_id = pair
            word1_freq = word_frequencies.get(word1_id, 0)
            word2_freq = word_frequencies.get(word2_id, 0)

            score = self.calculate_likelihood_score(pair, pair_freq, word1_freq, word2_freq)
            likelihood_scores[pair] = score

        # Update max tracking
        self._update_max_likelihood(likelihood_scores)

        return likelihood_scores

    def _update_max_likelihood(self, likelihood_scores: Dict[Tuple[int, int], float],
                               updated_pair: Tuple[int, int] = None):
        """
        Update max likelihood tracking efficiently.

        Args:
            likelihood_scores: Current likelihood score dictionary
            updated_pair: Specific pair that was updated (for targeted check)
        """
        if updated_pair and updated_pair in likelihood_scores:
            score = likelihood_scores[updated_pair]
            if score > self.max_likelihood:
                self.max_likelihood = score
                self.max_pair = updated_pair
        else:
            # Full scan only when necessary (initialization or max pair removed)
            if likelihood_scores:
                max_item = max(likelihood_scores.items(), key=lambda x: x[1])
                self.max_pair = max_item[0]  # pair
                self.max_likelihood = max_item[1]  # score
            else:
                self.max_likelihood = -float('inf')
                self.max_pair = None

    def find_best_pair_by_likelihood(self, likelihood_scores: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """
        Find the pair with highest likelihood score above threshold.

        Args:
            likelihood_scores: Dictionary of likelihood scores

        Returns:
            Best pair as (word1_id, word2_id) tuple, or None if no valid pairs
        """
        if not likelihood_scores or self.max_pair is None:
            return None

        if self.max_likelihood >= self.min_likelihood_score:
            return self.max_pair
        else:
            return None

    def update_inverted_index_after_merge(self, counterized_data: List[List[int]],
                                          modified_indices: List[int],
                                          merged_pair: Tuple[int, int],
                                          new_id: int) -> None:
        """
        Update inverted index after merging operation.

        Args:
            counterized_data: Updated documents after merge
            modified_indices: Indices of documents that were modified
            merged_pair: The (word1, word2) pair that was merged
            new_id: New ID created from the merge
        """
        word1, word2 = merged_pair

        # Update index for modified documents only
        for doc_idx in modified_indices:
            document = counterized_data[doc_idx]
            doc_words = set(document)

            # Remove this document from word1 and word2 if they no longer exist
            if word1 not in doc_words and word1 in self.inverted_index:
                self.inverted_index[word1].discard(doc_idx)
                if not self.inverted_index[word1]:
                    del self.inverted_index[word1]

            if word2 not in doc_words and word2 in self.inverted_index:
                self.inverted_index[word2].discard(doc_idx)
                if not self.inverted_index[word2]:
                    del self.inverted_index[word2]

            # Add new_id to index
            if new_id not in self.inverted_index:
                self.inverted_index[new_id] = set()
            self.inverted_index[new_id].add(doc_idx)

    def merge_word_pairs_vectorized(self, counterized_data: List[List[int]],
                                   pair_to_merge: Tuple[int, int], new_id: int,
                                   candidate_doc_indices: set = None) -> Tuple[List[List[int]], List[int]]:
        """
        Optimized vectorized merge with skip filtering and in-place modification.

        Args:
            counterized_data: List of documents with word IDs
            pair_to_merge: The (word1_id, word2_id) pair to replace
            new_id: New ID to replace the pair with
            candidate_doc_indices: Optional set of document indices to check (from inverted index)

        Returns:
            Tuple of (updated counterized data with pairs merged, list of modified document indices)
        """
        word1, word2 = pair_to_merge
        modified_indices = []

        # If candidate indices provided, only iterate over those documents
        if candidate_doc_indices is not None:
            doc_indices_to_check = candidate_doc_indices
        else:
            doc_indices_to_check = range(len(counterized_data))

        for doc_idx in doc_indices_to_check:
            doc_array = counterized_data[doc_idx]

            if len(doc_array) < 2:
                continue

            # Skip filtering only needed if we're checking all documents
            if candidate_doc_indices is None:
                if word1 not in doc_array or word2 not in doc_array:
                    continue

            # Find positions where consecutive pairs match
            matches = (doc_array[:-1] == word1) & (doc_array[1:] == word2)

            if not matches.any():
                continue

            # Track that this document was modified
            modified_indices.append(doc_idx)

            # Optimized vectorized merge
            match_positions = np.where(matches)[0]

            # Build result array by collecting segments
            result_segments = []
            last_end = 0

            for match_pos in match_positions:
                # Add elements before this match
                if match_pos > last_end:
                    result_segments.append(doc_array[last_end:match_pos])

                # Add merged token
                result_segments.append(np.array([new_id], dtype=doc_array.dtype))

                # Skip the pair (next segment starts after both elements)
                last_end = match_pos + 2

            # Add remaining elements after last match
            if last_end < len(doc_array):
                result_segments.append(doc_array[last_end:])

            # Concatenate all segments efficiently
            if result_segments:
                new_doc = np.concatenate(result_segments)
            else:
                new_doc = np.array([], dtype=doc_array.dtype)

            # In-place modification: replace document in original list
            counterized_data[doc_idx] = new_doc

        return counterized_data, modified_indices

    def update_frequencies_and_scores_incremental(self,
                                                 pair_frequencies: Dict[Tuple[int, int], int],
                                                 word_frequencies: Dict[int, int],
                                                 likelihood_scores: Dict[Tuple[int, int], float],
                                                 counterized_data: List[List[int]],
                                                 modified_indices: List[int],
                                                 old_documents: Dict[int, np.ndarray],
                                                 merged_pair: Tuple[int, int],
                                                 new_id: int) -> Tuple[Dict, Dict, Dict]:
        """
        Incrementally update frequencies and likelihood scores for modified documents.

        Args:
            pair_frequencies: Current pair frequency table
            word_frequencies: Current word frequency table
            likelihood_scores: Current likelihood scores
            counterized_data: Updated counterized data (after merge)
            modified_indices: Indices of documents that were modified
            old_documents: Dict mapping doc_idx -> original document before merge
            merged_pair: The pair that was merged
            new_id: New ID created from merge

        Returns:
            Tuple of (updated pair_frequencies, updated word_frequencies, updated likelihood_scores)
        """
        max_pair_removed = False
        pairs_to_update = set()  # Track pairs whose scores need recalculation

        word1, word2 = merged_pair

        for doc_idx in modified_indices:
            # Remove old frequencies from this document
            old_doc_array = old_documents[doc_idx]

            # Remove old word frequencies
            if NUMBA_AVAILABLE:
                old_words = count_words_numba(old_doc_array)
                for word_id in old_words:
                    if word_id in word_frequencies:
                        word_frequencies[word_id] -= 1
                        if word_frequencies[word_id] <= 0:
                            del word_frequencies[word_id]
            else:
                for word_id in old_doc_array:
                    if word_id in word_frequencies:
                        word_frequencies[word_id] -= 1
                        if word_frequencies[word_id] <= 0:
                            del word_frequencies[word_id]

            # Remove old pair frequencies
            if len(old_doc_array) >= 2:
                if NUMBA_AVAILABLE:
                    old_pairs_array = count_pairs_numba(old_doc_array)
                    for i in range(len(old_pairs_array)):
                        pair = (old_pairs_array[i, 0], old_pairs_array[i, 1])
                        if pair in pair_frequencies:
                            pair_frequencies[pair] -= 1
                            pairs_to_update.add(pair)
                            if pair_frequencies[pair] <= 0:
                                if pair == self.max_pair:
                                    max_pair_removed = True
                                del pair_frequencies[pair]
                                if pair in likelihood_scores:
                                    del likelihood_scores[pair]
                else:
                    old_pairs = np.column_stack([old_doc_array[:-1], old_doc_array[1:]])
                    for i in range(len(old_pairs)):
                        pair = (old_pairs[i, 0], old_pairs[i, 1])
                        if pair in pair_frequencies:
                            pair_frequencies[pair] -= 1
                            pairs_to_update.add(pair)
                            if pair_frequencies[pair] <= 0:
                                if pair == self.max_pair:
                                    max_pair_removed = True
                                del pair_frequencies[pair]
                                if pair in likelihood_scores:
                                    del likelihood_scores[pair]

            # Add new frequencies from this document
            new_doc_array = counterized_data[doc_idx]

            # Add new word frequencies
            if NUMBA_AVAILABLE:
                new_words = count_words_numba(new_doc_array)
                for word_id in new_words:
                    word_frequencies[word_id] = word_frequencies.get(word_id, 0) + 1
            else:
                for word_id in new_doc_array:
                    word_frequencies[word_id] = word_frequencies.get(word_id, 0) + 1

            # Add new pair frequencies
            if len(new_doc_array) >= 2:
                if NUMBA_AVAILABLE:
                    new_pairs_array = count_pairs_numba(new_doc_array)
                    for i in range(len(new_pairs_array)):
                        pair = (new_pairs_array[i, 0], new_pairs_array[i, 1])
                        pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1
                        pairs_to_update.add(pair)
                else:
                    new_pairs = np.column_stack([new_doc_array[:-1], new_doc_array[1:]])
                    for i in range(len(new_pairs)):
                        pair = (new_pairs[i, 0], new_pairs[i, 1])
                        pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1
                        pairs_to_update.add(pair)

        # Recalculate likelihood scores for affected pairs
        for pair in pairs_to_update:
            if pair in pair_frequencies:
                word1_id, word2_id = pair
                word1_freq = word_frequencies.get(word1_id, 0)
                word2_freq = word_frequencies.get(word2_id, 0)
                pair_freq = pair_frequencies[pair]

                score = self.calculate_likelihood_score(pair, pair_freq, word1_freq, word2_freq)
                likelihood_scores[pair] = score

                # Check if this is a new max
                if score > self.max_likelihood:
                    self.max_likelihood = score
                    self.max_pair = pair

        # If max pair was removed, need to find new max
        if max_pair_removed:
            self._update_max_likelihood(likelihood_scores)

        return pair_frequencies, word_frequencies, likelihood_scores

    def fit_optimized(self, counterized_data: List[List[int]], original_vocab_size: int,
                     original_vocab: List[str] = None) -> List[List[int]]:
        """
        Optimized training method using vectorized operations and likelihood-based selection.

        Args:
            counterized_data: Original counterized documents
            original_vocab_size: Size of the original vocabulary
            original_vocab: Original vocabulary for decoding (optional)

        Returns:
            Updated counterized data with n-gram merges applied
        """
        self.original_vocab_size = original_vocab_size
        self.current_vocab_size = original_vocab_size

        # Determine optimal dtype for memory efficiency
        dtype = self._get_optimal_dtype()

        # Work with numpy arrays for faster operations
        working_data = [np.array(doc, dtype=dtype) for doc in counterized_data]

        # Performance tracking
        start_time = time.time()
        timing_stats = {
            'inverted_index_time': 0,
            'word_freq_time': 0,
            'pair_freq_time': 0,
            'likelihood_calc_time': 0,
            'merge_time': 0,
            'update_time': 0,
            'total_iterations': 0
        }

        self._console.print_debug(f"Starting optimized WordPiece n-gram with vocab size: {self.current_vocab_size}", tag="WORDPIECE")
        self._console.print_debug(f"Target vocab limit: {self.vocab_limit}", tag="WORDPIECE")
        self._console.print_debug(f"Using dtype: {dtype.__name__} for memory efficiency", tag="WORDPIECE")
        self._console.print_debug("Using likelihood-based scoring with inverted index and vectorized operations", tag="WORDPIECE")

        # Build inverted index once at start
        self._console.print_debug("Building inverted index...", tag="WORDPIECE")
        idx_start = time.time()
        self.inverted_index = self.build_inverted_index(working_data)
        timing_stats['inverted_index_time'] = time.time() - idx_start
        self._console.print_debug(f"Inverted index built with {len(self.inverted_index)} unique tokens", tag="WORDPIECE")

        # Calculate initial word frequencies
        self._console.print_debug("Calculating word frequencies...", tag="WORDPIECE")
        word_freq_start = time.time()
        self.word_frequencies = self.calculate_word_frequencies(working_data)
        timing_stats['word_freq_time'] = time.time() - word_freq_start

        # Build initial pair frequency table
        self._console.print_debug("Building pair frequency table...", tag="WORDPIECE")
        pair_freq_start = time.time()
        self.pair_frequencies = self.build_pair_frequency_table(working_data)
        timing_stats['pair_freq_time'] = time.time() - pair_freq_start

        # Calculate initial likelihood scores
        self._console.print_debug("Calculating likelihood scores...", tag="WORDPIECE")
        likelihood_start = time.time()
        self.likelihood_scores = self.calculate_all_likelihood_scores(
            self.pair_frequencies, self.word_frequencies
        )
        timing_stats['likelihood_calc_time'] = time.time() - likelihood_start
        self._console.print_debug(f"Initial likelihood scores calculated for {len(self.likelihood_scores)} pairs", tag="WORDPIECE")

        # Create progress bar with rate display
        max_iterations = self.vocab_limit - self.current_vocab_size
        pbar = tqdm(total=max_iterations, desc="WordPiece N-gram",
                   unit="merge", disable=False, smoothing=0.3, mininterval=0.1)

        iteration = 0
        while self.current_vocab_size < self.vocab_limit:
            # Find best pair by likelihood score
            best_pair = self.find_best_pair_by_likelihood(self.likelihood_scores)

            if best_pair is None:
                pbar.close()
                self._console.print_debug(f"No more pairs meet minimum likelihood threshold. Stopping at vocab size: {self.current_vocab_size}", tag="WORDPIECE")
                break

            # Assign new ID to this pair
            new_id = self.current_vocab_size
            likelihood_score = self.likelihood_scores[best_pair]
            frequency = self.pair_frequencies[best_pair]
            word1, word2 = best_pair

            # Find candidate documents using inverted index
            candidate_docs = None
            if word1 in self.inverted_index and word2 in self.inverted_index:
                set1 = self.inverted_index[word1]
                set2 = self.inverted_index[word2]

                if not set1 or not set2:
                    candidate_docs = set()
                else:
                    if len(set1) <= len(set2):
                        candidate_docs = set1 & set2
                    else:
                        candidate_docs = set2 & set1

            # Decode pair for human-readable output (only if verbose)
            if self.verbose and original_vocab:
                token1_id, token2_id = best_pair

                if token1_id < self.original_vocab_size:
                    token1_text = original_vocab[token1_id] if token1_id < len(original_vocab) else f"UNK_{token1_id}"
                else:
                    token1_text = self.reconstruct_ngram_meaning(token1_id, original_vocab)

                if token2_id < self.original_vocab_size:
                    token2_text = original_vocab[token2_id] if token2_id < len(original_vocab) else f"UNK_{token2_id}"
                else:
                    token2_text = self.reconstruct_ngram_meaning(token2_id, original_vocab)

                tqdm.write(f"Iteration {iteration + 1}: Merging '{token1_text}'+'{token2_text}' "
                          f"(likelihood: {likelihood_score:.6f}, freq: {frequency}) -> ID {new_id}")

            # Batch progress bar updates
            if iteration % 25 == 0 or iteration < 10:
                pbar.set_postfix({'vocab': self.current_vocab_size, 'score': f'{likelihood_score:.6f}'})

            # Store merge operation with likelihood score and frequency
            self.merge_operations.append((best_pair, new_id, likelihood_score, frequency))
            self.pair_to_id[best_pair] = new_id
            self.id_to_pair[new_id] = best_pair

            # Backup documents that will be modified
            old_docs_backup = {}
            if candidate_docs:
                for doc_idx in candidate_docs:
                    doc_array = working_data[doc_idx]
                    if len(doc_array) >= 2:
                        matches = (doc_array[:-1] == word1) & (doc_array[1:] == word2)
                        if matches.any():
                            old_docs_backup[doc_idx] = working_data[doc_idx].copy()

            # Apply merge using vectorized operations
            merge_start = time.time()
            working_data, modified_indices = self.merge_word_pairs_vectorized(
                working_data, best_pair, new_id, candidate_doc_indices=candidate_docs
            )

            # Update inverted index after merge
            self.update_inverted_index_after_merge(working_data, modified_indices, best_pair, new_id)
            timing_stats['merge_time'] += time.time() - merge_start

            # Incrementally update frequencies and likelihood scores
            update_start = time.time()
            self.pair_frequencies, self.word_frequencies, self.likelihood_scores = \
                self.update_frequencies_and_scores_incremental(
                    self.pair_frequencies, self.word_frequencies, self.likelihood_scores,
                    working_data, modified_indices, old_docs_backup, best_pair, new_id
                )
            timing_stats['update_time'] += time.time() - update_start

            # Update vocabulary size
            self.current_vocab_size += 1
            iteration += 1

            # Batch progress bar updates
            if iteration % 25 == 0 or iteration < 10:
                pbar.update(25 if iteration >= 25 else 1)
            elif self.current_vocab_size >= self.vocab_limit:
                pbar.update(iteration % 25)

            # Safety check to prevent infinite loops
            if iteration > 50000:
                pbar.close()
                self._console.print_warning("Maximum iterations reached. Stopping merge process.", tag="WORDPIECE")
                break

        pbar.close()

        # Performance report
        total_time = time.time() - start_time
        timing_stats['total_iterations'] = iteration

        self._console.print_debug(f"Optimized WordPiece n-gram completed. Final vocab size: {self.current_vocab_size}", tag="WORDPIECE")
        self._console.print_debug(f"Created {len(self.merge_operations)} n-gram combinations", tag="WORDPIECE")
        self._console.print_debug("Performance Report:", tag="WORDPIECE")
        self._console.print_debug(f"  Total time: {total_time:.2f}s", tag="WORDPIECE")
        self._console.print_debug(f"  Inverted index build: {timing_stats['inverted_index_time']:.2f}s", tag="WORDPIECE")
        self._console.print_debug(f"  Word frequency calculation: {timing_stats['word_freq_time']:.2f}s", tag="WORDPIECE")
        self._console.print_debug(f"  Pair frequency calculation: {timing_stats['pair_freq_time']:.2f}s", tag="WORDPIECE")
        self._console.print_debug(f"  Initial likelihood calculation: {timing_stats['likelihood_calc_time']:.2f}s", tag="WORDPIECE")
        self._console.print_debug(f"  Merge operations: {timing_stats['merge_time']:.2f}s ({timing_stats['merge_time']/total_time*100:.1f}%)", tag="WORDPIECE")
        self._console.print_debug(f"  Frequency/score updates: {timing_stats['update_time']:.2f}s ({timing_stats['update_time']/total_time*100:.1f}%)", tag="WORDPIECE")
        if iteration > 0:
            self._console.print_debug(f"  Average per iteration: {total_time/iteration*1000:.1f}ms", tag="WORDPIECE")
            self._console.print_debug(f"  Iterations per second: {iteration/total_time:.1f}", tag="WORDPIECE")

        return working_data

    def fit(self, counterized_data: List[List[int]], original_vocab_size: int,
            original_vocab: List[str] = None) -> List[List[int]]:
        """
        Basic training method using likelihood-based selection.

        Args:
            counterized_data: Original counterized documents
            original_vocab_size: Size of the original vocabulary
            original_vocab: Original vocabulary for decoding (optional)

        Returns:
            Updated counterized data with n-gram merges applied
        """
        self.original_vocab_size = original_vocab_size
        self.current_vocab_size = original_vocab_size

        # Determine optimal dtype
        dtype = self._get_optimal_dtype()
        working_data = [np.array(doc, dtype=dtype) for doc in counterized_data]

        self._console.print_debug(f"Starting WordPiece n-gram with vocab size: {self.current_vocab_size}", tag="WORDPIECE")
        self._console.print_debug(f"Target vocab limit: {self.vocab_limit}", tag="WORDPIECE")
        self._console.print_debug(f"Using dtype: {dtype.__name__} for memory efficiency", tag="WORDPIECE")

        # Create progress bar
        max_iterations = self.vocab_limit - self.current_vocab_size
        pbar = tqdm(total=max_iterations, desc="WordPiece N-gram",
                   unit="merge", disable=False, smoothing=0.3, mininterval=0.1)

        iteration = 0
        while self.current_vocab_size < self.vocab_limit:
            # Calculate frequencies
            self.word_frequencies = self.calculate_word_frequencies(working_data)
            self.pair_frequencies = self.build_pair_frequency_table(working_data)

            # Calculate likelihood scores
            self.likelihood_scores = self.calculate_all_likelihood_scores(
                self.pair_frequencies, self.word_frequencies
            )

            # Find best pair by likelihood
            best_pair = self.find_best_pair_by_likelihood(self.likelihood_scores)

            if best_pair is None:
                pbar.close()
                self._console.print_debug(f"No more pairs meet minimum likelihood threshold. Stopping at vocab size: {self.current_vocab_size}", tag="WORDPIECE")
                break

            # Assign new ID
            new_id = self.current_vocab_size
            likelihood_score = self.likelihood_scores[best_pair]
            frequency = self.pair_frequencies[best_pair]

            # Update progress bar
            pbar.set_postfix({'vocab': self.current_vocab_size, 'score': f'{likelihood_score:.6f}'})

            # Decode pair for output (only if verbose)
            if self.verbose and original_vocab:
                token1_id, token2_id = best_pair

                if token1_id < self.original_vocab_size:
                    token1_text = original_vocab[token1_id] if token1_id < len(original_vocab) else f"UNK_{token1_id}"
                else:
                    token1_text = self.reconstruct_ngram_meaning(token1_id, original_vocab)

                if token2_id < self.original_vocab_size:
                    token2_text = original_vocab[token2_id] if token2_id < len(original_vocab) else f"UNK_{token2_id}"
                else:
                    token2_text = self.reconstruct_ngram_meaning(token2_id, original_vocab)

                tqdm.write(f"Iteration {iteration + 1}: Merging '{token1_text}'+'{token2_text}' "
                          f"(likelihood: {likelihood_score:.6f}, freq: {frequency}) -> ID {new_id}")

            # Store merge operation
            self.merge_operations.append((best_pair, new_id, likelihood_score, frequency))
            self.pair_to_id[best_pair] = new_id
            self.id_to_pair[new_id] = best_pair

            # Apply merge
            working_data, _ = self.merge_word_pairs_vectorized(working_data, best_pair, new_id)

            # Update vocabulary size
            self.current_vocab_size += 1
            iteration += 1
            pbar.update(1)

            # Safety check
            if iteration > 50000:
                pbar.close()
                self._console.print_warning("Maximum iterations reached. Stopping merge process.", tag="WORDPIECE")
                break

        pbar.close()
        self._console.print_debug(f"WordPiece n-gram completed. Final vocab size: {self.current_vocab_size}", tag="WORDPIECE")
        self._console.print_debug(f"Created {len(self.merge_operations)} n-gram combinations", tag="WORDPIECE")

        return working_data

    def get_ngram_vocab_info(self) -> Dict:
        """
        Return information about the n-gram vocabulary created.

        Returns:
            Dictionary containing vocabulary statistics and mappings
        """
        return {
            'original_vocab_size': self.original_vocab_size,
            'final_vocab_size': self.current_vocab_size,
            'ngrams_created': len(self.merge_operations),
            'merge_operations': self.merge_operations[:10],  # Show first 10 for inspection
            'pair_to_id_sample': dict(list(self.pair_to_id.items())[:5])  # Sample mappings
        }

    def reconstruct_ngram_meaning(self, ngram_id: int, original_vocab: List[str]) -> str:
        """
        Reconstruct the meaning of an n-gram ID back to original words.

        Args:
            ngram_id: ID of the n-gram to reconstruct
            original_vocab: Original vocabulary list (word index -> word string)

        Returns:
            String representation of the n-gram
        """
        if ngram_id < self.original_vocab_size:
            # This is an original word
            return original_vocab[ngram_id] if ngram_id < len(original_vocab) else f"UNK_{ngram_id}"

        if ngram_id not in self.id_to_pair:
            return f"NGRAM_{ngram_id}"

        # Recursively reconstruct the pair
        word1_id, word2_id = self.id_to_pair[ngram_id]
        word1_str = self.reconstruct_ngram_meaning(word1_id, original_vocab)
        word2_str = self.reconstruct_ngram_meaning(word2_id, original_vocab)

        return f"{word1_str}_{word2_str}"

    def save_ngrams_to_json(self, filename: str, original_vocab: List[str], output_dir: str = None) -> str:
        """
        Save n-gram pairs and their meanings to a JSON file.

        Args:
            filename: Name of the JSON file to save
            original_vocab: Original vocabulary list for decoding
            output_dir: Directory to save the file (optional)

        Returns:
            Full path of the saved file
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename

        ngrams_data = {
            "metadata": {
                "algorithm": "WordPiece",
                "original_vocab_size": self.original_vocab_size,
                "final_vocab_size": self.current_vocab_size,
                "ngrams_created": len(self.merge_operations),
                "vocab_limit": self.vocab_limit,
                "min_likelihood_score": self.min_likelihood_score,
                "smoothing": self.smoothing
            },
            "ngrams": {}
        }

        # Export each n-gram with its information
        for pair, new_id, likelihood_score, frequency in self.merge_operations:
            token1_id, token2_id = pair

            # Decode tokens to readable text (handle both original and n-gram tokens)
            if token1_id < self.original_vocab_size:
                token1_text = original_vocab[token1_id] if token1_id < len(original_vocab) else f"UNK_{token1_id}"
            else:
                token1_text = self.reconstruct_ngram_meaning(token1_id, original_vocab)

            if token2_id < self.original_vocab_size:
                token2_text = original_vocab[token2_id] if token2_id < len(original_vocab) else f"UNK_{token2_id}"
            else:
                token2_text = self.reconstruct_ngram_meaning(token2_id, original_vocab)

            ngrams_data["ngrams"][str(new_id)] = {
                "pair": f"{token1_text},{token2_text}",
                "likelihood_score": likelihood_score,
                "frequency": frequency
            }

        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ngrams_data, f, ensure_ascii=False, indent=2)

        self._console.print_status(f"N-grams saved to: {filepath}", "success")
        return filepath
