"""
N-gram BPE (Byte Pair Encoding) implementation for word-level merging.

This module implements a BPE-like algorithm that operates on counterized (numerical)
word data instead of character-level text. It merges frequent adjacent word pairs
to create meaningful n-gram combinations while maintaining memory efficiency.

Example:
    If we have words "good" (ID=1) and "product" (ID=2), and they frequently
    appear together, the algorithm will create a new combined token (ID=3)
    representing the "good product" n-gram.
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from itertools import chain
from operator import itemgetter
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

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
        2D array where each row is [word1, word2, count=1]
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


class WordPairBPE:
    """
    BPE-like algorithm for creating n-gram word pairs from counterized data.

    This class implements a memory-efficient approach to generating meaningful
    word n-grams by iteratively merging the most frequent adjacent word pairs
    until a vocabulary size limit is reached.
    """

    __slots__ = (
        'vocab_limit', 'min_pair_frequency', 'verbose', 'use_pmi',
        'original_vocab_size', 'current_vocab_size', 'merge_operations',
        'pair_to_id', 'id_to_pair', 'pair_frequencies', 'inverted_index',
        'max_frequency', 'max_pair', 'token_frequencies', 'total_tokens',
        '_ngram_cache', '_console'
    )

    def __init__(
        self,
        vocab_limit: int = 10000,
        min_pair_frequency: int = 2,
        verbose: bool = False,
        use_pmi: bool = False,
        console: Optional["ConsoleManager"] = None,
    ):
        """
        Initialize the WordPairBPE encoder.

        Args:
            vocab_limit: Maximum vocabulary size before stopping merging
            min_pair_frequency: Minimum frequency threshold for pair merging
            verbose: Whether to print detailed decoding information (default: False for speed)
            use_pmi: Use PMI (Pointwise Mutual Information) scoring instead of raw frequency.
                    This helps number-word pairs compete fairly with word-word pairs.
            console: Console manager for output. If None, uses global console.
        """
        # Import here to avoid circular imports
        from ...utils.console.console_manager import get_console
        self._console = console or get_console()

        self.vocab_limit = vocab_limit
        self.min_pair_frequency = min_pair_frequency
        self.verbose = verbose
        self.use_pmi = use_pmi
        self.original_vocab_size = 0
        self.current_vocab_size = 0
        self.merge_operations = []  # List of (pair, new_id, frequency) tuples
        self.pair_to_id = {}  # Mapping of merged pairs to their new IDs
        self.id_to_pair = {}  # Reverse mapping for reconstruction
        self.pair_frequencies = {}  # Store frequencies for each merged pair
        self.inverted_index = {}  # Mapping of word_id -> set of doc indices containing that word

        # Optimization: Track max frequency pair for O(1) lookup
        self.max_frequency = 0
        self.max_pair = None

        # PMI scoring state
        self.token_frequencies = {}  # Token marginal frequencies for PMI
        self.total_tokens = 0  # Total token count for PMI

        # Memoization cache for reconstruct_ngram_meaning
        self._ngram_cache = {}

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

    def build_token_frequencies(
        self, counterized_data: List[List[int]]
    ) -> Tuple[Dict[int, int], int]:
        """
        Compute marginal frequency of each token across all documents.

        This is used for PMI scoring to compute P(x) and P(y) probabilities.

        Args:
            counterized_data: List of documents with word IDs

        Returns:
            Tuple of (token_freq_dict, total_token_count)
        """
        # Flatten all documents into single iterable (memory efficient)
        all_tokens = chain.from_iterable(counterized_data)

        # Use Counter for efficient counting - it's optimized in C
        token_counter = Counter(int(t) for t in all_tokens)

        return dict(token_counter), sum(token_counter.values())

    def compute_pmi(self, pair: Tuple[int, int], pair_freq: int, total_pairs: int) -> float:
        """
        Compute PMI (Pointwise Mutual Information) score for a token pair.

        PMI(x, y) = log2(P(x,y) / (P(x) * P(y)))

        Higher PMI means the pair appears together more than expected by chance.
        This helps number-word pairs compete fairly with word-word pairs.

        Args:
            pair: (token1_id, token2_id) tuple
            pair_freq: Frequency of this pair
            total_pairs: Total number of pairs in corpus

        Returns:
            PMI score (higher = stronger association)
        """
        token1, token2 = pair
        freq1 = self.token_frequencies.get(int(token1), 1)
        freq2 = self.token_frequencies.get(int(token2), 1)

        # P(x,y), P(x), P(y)
        p_xy = pair_freq / total_pairs if total_pairs > 0 else 0
        p_x = freq1 / self.total_tokens if self.total_tokens > 0 else 0
        p_y = freq2 / self.total_tokens if self.total_tokens > 0 else 0

        # PMI = log2(P(x,y) / (P(x) * P(y)))
        denominator = p_x * p_y
        if denominator > 0 and p_xy > 0:
            pmi = math.log2(p_xy / denominator)
        else:
            pmi = 0.0

        return pmi

    def build_pair_frequency_table_optimized(self, counterized_data: List[List[int]]) -> dict:
        """
        Optimized version using numpy vectorized operations and Numba JIT.
        Returns plain dict instead of Counter for better performance.

        Args:
            counterized_data: List of documents, each containing word IDs or numpy arrays

        Returns:
            Dict object with (word1_id, word2_id) -> frequency mappings
        """
        if NUMBA_AVAILABLE:
            # Use Numba-optimized batch pair extraction
            all_pairs = count_pairs_from_docs_numba(counterized_data)

            # Use map + Counter for batch counting (C-optimized)
            pair_frequencies = dict(Counter(map(tuple, all_pairs)))

            # Initialize max tracking
            self._update_max_frequency(pair_frequencies)
            return pair_frequencies
        else:
            # Fallback using generator + Counter (memory efficient)
            def pair_generator():
                for doc_array in counterized_data:
                    if len(doc_array) >= 2:
                        # Use zip for memory-efficient pairing
                        yield from zip(doc_array[:-1].tolist(), doc_array[1:].tolist())

            pair_frequencies = dict(Counter(pair_generator()))

            # Initialize max tracking
            self._update_max_frequency(pair_frequencies)
            return pair_frequencies

    def update_inverted_index_after_merge(
        self,
        counterized_data: List[List[int]],
        modified_indices: List[int],
        merged_pair: Tuple[int, int],
        new_id: int,
    ) -> None:
        """
        Update inverted index after merging operation.

        Args:
            counterized_data: Updated documents after merge
            modified_indices: Indices of documents that were modified
            merged_pair: The (word1, word2) pair that was merged
            new_id: New ID created from the merge
        """
        word1, word2 = merged_pair
        inv_idx = self.inverted_index  # Local ref for faster access

        # Pre-create set for new_id if needed (avoid repeated 'in' checks)
        if new_id not in inv_idx:
            inv_idx[new_id] = set()
        new_id_set = inv_idx[new_id]

        # Update index for modified documents only
        for doc_idx in modified_indices:
            doc_words = set(counterized_data[doc_idx])

            # Remove this document from word1 and word2 if they no longer exist
            # Use dict.get() with walrus operator for cleaner conditional cleanup
            for word in (word1, word2):
                if word not in doc_words and (word_set := inv_idx.get(word)):
                    word_set.discard(doc_idx)
                    if not word_set:
                        del inv_idx[word]

            new_id_set.add(doc_idx)

    def merge_word_pairs_vectorized(
        self,
        counterized_data: List[List[int]],
        pair_to_merge: Tuple[int, int],
        new_id: int,
        candidate_doc_indices: set = None,
    ) -> Tuple[List[List[int]], List[int]]:
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

            # Optimized vectorized merge - safer approach without skip mask
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
    
    def update_pair_frequencies_incremental(
        self,
        pair_frequencies: dict,
        counterized_data: List[List[int]],
        modified_indices: List[int],
        old_documents: Dict[int, np.ndarray],
    ) -> dict:
        """
        Incrementally update pair frequencies for modified documents only.
        Optimized to use plain dict and update max tracking efficiently.

        Args:
            pair_frequencies: Current pair frequency table
            counterized_data: Updated counterized data (after merge)
            modified_indices: Indices of documents that were modified
            old_documents: Dict mapping doc_idx -> original document before merge

        Returns:
            Updated dict with incremental frequency changes
        """
        # Work directly on the original dict for better performance
        updated_frequencies = pair_frequencies
        max_pair_removed = False

        for doc_idx in modified_indices:
            # Remove old pairs from this document using optimized extraction
            old_doc_array = old_documents[doc_idx]
            if len(old_doc_array) >= 2:
                if NUMBA_AVAILABLE:
                    old_pairs_array = count_pairs_numba(old_doc_array)
                    for i in range(len(old_pairs_array)):
                        pair = (old_pairs_array[i, 0], old_pairs_array[i, 1])
                        if pair in updated_frequencies:
                            updated_frequencies[pair] -= 1
                            if updated_frequencies[pair] <= 0:
                                if pair == self.max_pair:
                                    max_pair_removed = True
                                del updated_frequencies[pair]
                else:
                    # Fallback: vectorized numpy without list conversion
                    old_pairs = np.column_stack([old_doc_array[:-1], old_doc_array[1:]])
                    for i in range(len(old_pairs)):
                        pair = (old_pairs[i, 0], old_pairs[i, 1])
                        if pair in updated_frequencies:
                            updated_frequencies[pair] -= 1
                            if updated_frequencies[pair] <= 0:
                                if pair == self.max_pair:
                                    max_pair_removed = True
                                del updated_frequencies[pair]

            # Add new pairs from this document using optimized extraction
            new_doc_array = counterized_data[doc_idx]
            if len(new_doc_array) >= 2:
                if NUMBA_AVAILABLE:
                    new_pairs_array = count_pairs_numba(new_doc_array)
                    for i in range(len(new_pairs_array)):
                        pair = (new_pairs_array[i, 0], new_pairs_array[i, 1])
                        updated_frequencies[pair] = updated_frequencies.get(pair, 0) + 1
                        # Check if this is a new max
                        self._update_max_frequency(updated_frequencies, pair)
                else:
                    # Fallback: vectorized numpy without list conversion
                    new_pairs = np.column_stack([new_doc_array[:-1], new_doc_array[1:]])
                    for i in range(len(new_pairs)):
                        pair = (new_pairs[i, 0], new_pairs[i, 1])
                        updated_frequencies[pair] = updated_frequencies.get(pair, 0) + 1
                        # Check if this is a new max
                        self._update_max_frequency(updated_frequencies, pair)

        # If max pair was removed, need to find new max
        if max_pair_removed:
            self._update_max_frequency(updated_frequencies)

        return updated_frequencies

    def build_pair_frequency_table(self, counterized_data: List[List[int]]) -> Counter:
        """
        Build frequency table of adjacent word pairs.

        Args:
            counterized_data: List of documents, each containing word IDs

        Returns:
            Counter object with (word1_id, word2_id) -> frequency mappings
        """
        # Single-pass with chain.from_iterable for memory efficiency
        all_pairs = chain.from_iterable(
            zip(doc[:-1], doc[1:]) for doc in counterized_data if len(doc) >= 2
        )

        # Counter.update() from iterable is C-optimized
        return Counter(all_pairs)

    def _update_max_frequency(self, pair_frequencies: dict, updated_pair: Tuple[int, int] = None):
        """
        Update max frequency tracking efficiently.

        Args:
            pair_frequencies: Current frequency dictionary
            updated_pair: Specific pair that was updated (for targeted check)
        """
        if updated_pair and updated_pair in pair_frequencies:
            freq = pair_frequencies[updated_pair]
            if freq > self.max_frequency:
                self.max_frequency = freq
                self.max_pair = updated_pair
        elif pair_frequencies:
            # Full scan only when necessary (initialization or max pair removed)
            # itemgetter(1) is ~20% faster than lambda x: x[1]
            self.max_pair, self.max_frequency = max(
                pair_frequencies.items(),
                key=itemgetter(1)
            )
        else:
            self.max_frequency = 0
            self.max_pair = None

    def find_most_frequent_pair(self, pair_frequencies: dict) -> Optional[Tuple[int, int]]:
        """
        Find the best word pair to merge based on scoring method.

        When use_pmi is True, uses PMI (Pointwise Mutual Information) scoring
        which helps number-word pairs compete fairly with word-word pairs.
        Otherwise uses O(1) max frequency tracking.

        Args:
            pair_frequencies: Dict with pair frequencies

        Returns:
            Best pair as (word1_id, word2_id) tuple, or None if no valid pairs
        """
        if not pair_frequencies:
            return None

        if self.use_pmi:
            # Score by weighted PMI: PMI * log2(freq + 1)
            # This balances association strength with occurrence frequency
            # Prevents rare-but-associated pairs from dominating
            total_pairs = sum(pair_frequencies.values())
            best_pair = None
            best_score = float("-inf")

            # Create a generator for pairs that meet the minimum frequency threshold.
            # This is more memory-efficient than creating an intermediate list.
            candidate_pairs = (
                (pair, freq)
                for pair, freq in pair_frequencies.items()
                if freq >= self.min_pair_frequency
            )

            # Use the `max` function with a key, which is generally faster and more
            # Pythonic than a manual loop for finding the maximum item.
            # The key function calculates the weighted PMI score for each candidate pair.
            # The `default=None` argument handles the case where candidate_pairs is empty.
            best_pair_item = max(
                candidate_pairs,
                key=lambda item: self.compute_pmi(item[0], item[1], total_pairs)
                * math.log2(item[1] + 1),
                default=None,
            )
            
            # Extract the pair from the result
            best_pair = best_pair_item[0] if best_pair_item else None

            return best_pair
        else:
            # Original frequency-based logic with O(1) max tracking
            if self.max_pair is None:
                return None

            if self.max_frequency >= self.min_pair_frequency:
                return self.max_pair
            else:
                return None
                
    def merge_word_pairs(
        self, counterized_data: List[List[int]], pair_to_merge: Tuple[int, int], new_id: int
    ) -> List[List[int]]:
        """
        Replace all occurrences of a word pair with a new combined ID.

        Args:
            counterized_data: List of documents with word IDs
            pair_to_merge: The (word1_id, word2_id) pair to replace
            new_id: New ID to replace the pair with

        Returns:
            Updated counterized data with pairs merged
        """
        word1, word2 = pair_to_merge
        updated_data = []
        
        for document in counterized_data:
            if len(document) < 2:
                updated_data.append(document[:])
                continue

            new_document = []
            i = 0

            while i < len(document):
                # Check if current and next word form the target pair
                if i < len(document) - 1 and document[i] == word1 and document[i + 1] == word2:
                    # Replace pair with new ID
                    new_document.append(new_id)
                    i += 2  # Skip both words
                else:
                    # Keep current word as-is
                    new_document.append(document[i])
                    i += 1

            updated_data.append(new_document)

        return updated_data

    def fit_optimized(
        self,
        counterized_data: List[List[int]],
        original_vocab_size: int,
        original_vocab: List[str] = None,
    ) -> List[List[int]]:
        """
        Optimized training method using vectorized operations and optimized data structures.

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

        # Work with numpy arrays for faster operations (no conversion overhead)
        working_data = [np.array(doc, dtype=dtype) for doc in counterized_data]

        # Performance tracking
        start_time = time.time()
        timing_stats = {
            "inverted_index_time": 0,
            "initial_freq_time": 0,
            "merge_time": 0,
            "freq_update_time": 0,
            "total_iterations": 0,
        }

        self._console.print_debug(f"Starting optimized n-gram BPE with vocab size: {self.current_vocab_size}", tag="N-GRAM BPE")
        self._console.print_debug(f"Target vocab limit: {self.vocab_limit}", tag="N-GRAM BPE")
        self._console.print_debug(f"Using dtype: {dtype.__name__} for memory efficiency", tag="N-GRAM BPE")
        if self.use_pmi:
            self._console.print_debug("Using PMI scoring (helps number-word pairs compete with word-word pairs)", tag="N-GRAM BPE")
        self._console.print_debug("Using inverted index, vectorized operations, and incremental frequency updates", tag="N-GRAM BPE")

        # Build inverted index once at start
        self._console.print_debug("Building inverted index...", tag="N-GRAM BPE")
        idx_start = time.time()
        self.inverted_index = self.build_inverted_index(working_data)
        timing_stats["inverted_index_time"] = time.time() - idx_start
        self._console.print_debug(f"Inverted index built with {len(self.inverted_index)} unique tokens", tag="N-GRAM BPE")

        # Build token frequencies for PMI scoring
        if self.use_pmi:
            self._console.print_debug("Computing token frequencies for PMI scoring...", tag="N-GRAM BPE")
            pmi_start = time.time()
            self.token_frequencies, self.total_tokens = self.build_token_frequencies(working_data)
            pmi_time = time.time() - pmi_start
            self._console.print_debug(
                f"Token frequencies computed in {pmi_time:.2f}s (total tokens: {self.total_tokens}, unique: {len(self.token_frequencies)})",
                tag="N-GRAM BPE"
            )

        # Build initial frequency table once
        freq_start = time.time()
        pair_frequencies = self.build_pair_frequency_table_optimized(working_data)
        timing_stats["initial_freq_time"] = time.time() - freq_start

        # Create progress bar with rate display
        max_iterations = self.vocab_limit - self.current_vocab_size
        pbar = tqdm(
            total=max_iterations,
            desc="N-gram BPE",
            unit="merge",
            disable=False,
            smoothing=0.3,
            mininterval=0.1,
        )

        iteration = 0
        while self.current_vocab_size < self.vocab_limit:
            # Find most frequent pair from current frequency table
            most_frequent_pair = self.find_most_frequent_pair(pair_frequencies)

            if most_frequent_pair is None:
                pbar.close()
                self._console.print_debug(
                    f"No more pairs meet minimum frequency threshold. Stopping at vocab size: {self.current_vocab_size}",
                    tag="N-GRAM BPE"
                )
                break

            # Assign new ID to this pair
            new_id = self.current_vocab_size
            frequency = pair_frequencies[most_frequent_pair]
            word1, word2 = most_frequent_pair

            # Find candidate documents using inverted index (intersection of docs containing word1 and word2)
            # Optimization: Check sizes before expensive set intersection
            candidate_docs = None
            if word1 in self.inverted_index and word2 in self.inverted_index:
                set1 = self.inverted_index[word1]
                set2 = self.inverted_index[word2]

                # Early exit if either set is empty
                if not set1 or not set2:
                    candidate_docs = set()
                else:
                    # Use smaller set for intersection to minimize operations
                    if len(set1) <= len(set2):
                        candidate_docs = set1 & set2
                    else:
                        candidate_docs = set2 & set1

            # Decode pair for human-readable output (only if verbose)
            if self.verbose and original_vocab:
                token1_id, token2_id = most_frequent_pair

                # Decode token1 (could be original or previously created n-gram)
                if token1_id < self.original_vocab_size:
                    token1_text = (
                        original_vocab[token1_id]
                        if token1_id < len(original_vocab)
                        else f"UNK_{token1_id}"
                    )
                else:
                    # This is a previously created n-gram, reconstruct it
                    token1_text = self.reconstruct_ngram_meaning(token1_id, original_vocab)

                # Decode token2 (could be original or previously created n-gram)
                if token2_id < self.original_vocab_size:
                    token2_text = (
                        original_vocab[token2_id]
                        if token2_id < len(original_vocab)
                        else f"UNK_{token2_id}"
                    )
                else:
                    # This is a previously created n-gram, reconstruct it
                    token2_text = self.reconstruct_ngram_meaning(token2_id, original_vocab)

                tqdm.write(
                    f"Iteration {iteration + 1}: Merging '{token1_text}'+'{token2_text}' "
                    f"(freq: {frequency}) -> ID {new_id}"
                )

            # Optimization: Batch progress bar updates to reduce I/O overhead
            if iteration % 25 == 0 or iteration < 10:  # Update frequently at start, then batch
                pbar.set_postfix({"vocab_size": self.current_vocab_size, "freq": frequency})

            # Store merge operation with frequency
            self.merge_operations.append((most_frequent_pair, new_id, frequency))
            self.pair_to_id[most_frequent_pair] = new_id
            self.id_to_pair[new_id] = most_frequent_pair
            self.pair_frequencies[most_frequent_pair] = frequency

            # Optimization: Only backup documents that actually contain the pair to merge
            if candidate_docs:
                word1, word2 = most_frequent_pair

                def has_pair(doc_idx):
                    doc_array = working_data[doc_idx]
                    return (len(doc_array) >= 2 and
                            ((doc_array[:-1] == word1) & (doc_array[1:] == word2)).any())

                old_docs_backup = {
                    doc_idx: working_data[doc_idx].copy()
                    for doc_idx in candidate_docs if has_pair(doc_idx)
                }
            else:
                old_docs_backup = {}

            # Apply merge using vectorized operations with candidate docs from inverted index
            merge_start = time.time()
            working_data, modified_indices = self.merge_word_pairs_vectorized(
                working_data, most_frequent_pair, new_id, candidate_doc_indices=candidate_docs
            )

            # Update inverted index after merge
            self.update_inverted_index_after_merge(
                working_data, modified_indices, most_frequent_pair, new_id
            )
            timing_stats["merge_time"] += time.time() - merge_start

            # Update token frequencies for PMI scoring
            if self.use_pmi and modified_indices:
                # Count actual merges by looking at document length changes
                total_merges = 0
                for doc_idx in modified_indices:
                    if doc_idx in old_docs_backup:
                        old_len = len(old_docs_backup[doc_idx])
                        new_len = len(working_data[doc_idx])
                        # Each merge reduces length by 1 (two tokens -> one)
                        merges_in_doc = old_len - new_len
                        total_merges += merges_in_doc

                if total_merges > 0:
                    # Decrease token frequencies for merged tokens
                    word1, word2 = most_frequent_pair
                    self.token_frequencies[int(word1)] = max(
                        0, self.token_frequencies.get(int(word1), 0) - total_merges
                    )
                    self.token_frequencies[int(word2)] = max(
                        0, self.token_frequencies.get(int(word2), 0) - total_merges
                    )
                    # Increase frequency for new merged token
                    self.token_frequencies[int(new_id)] = (
                        self.token_frequencies.get(int(new_id), 0) + total_merges
                    )
                    # Total tokens decreases (two tokens become one for each merge)
                    self.total_tokens -= total_merges

            # Incrementally update frequency table (only for modified documents)
            freq_update_start = time.time()
            pair_frequencies = self.update_pair_frequencies_incremental(
                pair_frequencies, working_data, modified_indices, old_docs_backup
            )
            timing_stats["freq_update_time"] += time.time() - freq_update_start

            # Update vocabulary size
            self.current_vocab_size += 1
            iteration += 1

            # Optimization: Batch progress bar updates
            if iteration % 25 == 0 or iteration < 10:
                pbar.update(25 if iteration >= 25 else 1)
            elif self.current_vocab_size >= self.vocab_limit:  # Always update at the end
                pbar.update(iteration % 25)

            # Safety check to prevent infinite loops
            if iteration > 50000:
                pbar.close()
                self._console.print_warning("Maximum iterations reached. Stopping merge process.", tag="N-GRAM BPE")
                break

        pbar.close()

        # Performance report
        total_time = time.time() - start_time
        timing_stats["total_iterations"] = iteration

        self._console.print_debug(f"Optimized n-gram BPE completed. Final vocab size: {self.current_vocab_size}", tag="N-GRAM BPE")
        self._console.print_debug(f"Created {len(self.merge_operations)} n-gram combinations", tag="N-GRAM BPE")
        self._console.print_debug("Performance Report:", tag="N-GRAM BPE")
        self._console.print_debug(f"  Total time: {total_time:.2f}s", tag="N-GRAM BPE")
        self._console.print_debug(f"  Inverted index build: {timing_stats['inverted_index_time']:.2f}s", tag="N-GRAM BPE")
        self._console.print_debug(f"  Initial frequency table: {timing_stats['initial_freq_time']:.2f}s", tag="N-GRAM BPE")
        self._console.print_debug(
            f"  Merge operations: {timing_stats['merge_time']:.2f}s ({timing_stats['merge_time'] / total_time * 100:.1f}%)",
            tag="N-GRAM BPE"
        )
        self._console.print_debug(
            f"  Frequency updates: {timing_stats['freq_update_time']:.2f}s ({timing_stats['freq_update_time'] / total_time * 100:.1f}%)",
            tag="N-GRAM BPE"
        )
        if iteration > 0:
            self._console.print_debug(f"  Average per iteration: {total_time / iteration * 1000:.1f}ms", tag="N-GRAM BPE")
            self._console.print_debug(f"  Iterations per second: {iteration / total_time:.1f}", tag="N-GRAM BPE")

        return working_data

    def fit(
        self,
        counterized_data: List[List[int]],
        original_vocab_size: int,
        original_vocab: List[str] = None,
    ) -> List[List[int]]:
        """
        Main training method that iteratively merges word pairs.

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

        # Work with a copy to avoid modifying original data
        working_data = [np.array(doc, dtype=dtype) for doc in counterized_data]

        self._console.print_debug(f"Starting n-gram BPE with vocab size: {self.current_vocab_size}", tag="N-GRAM BPE")
        self._console.print_debug(f"Target vocab limit: {self.vocab_limit}", tag="N-GRAM BPE")
        self._console.print_debug(f"Using dtype: {dtype.__name__} for memory efficiency", tag="N-GRAM BPE")

        # Create progress bar with rate display
        max_iterations = self.vocab_limit - self.current_vocab_size
        pbar = tqdm(
            total=max_iterations,
            desc="N-gram BPE",
            unit="merge",
            disable=False,
            smoothing=0.3,
            mininterval=0.1,
        )

        iteration = 0
        while self.current_vocab_size < self.vocab_limit:
            # Build frequency table for current iteration
            pair_frequencies = self.build_pair_frequency_table(working_data)

            # Find most frequent pair
            most_frequent_pair = self.find_most_frequent_pair(pair_frequencies)

            if most_frequent_pair is None:
                pbar.close()
                self._console.print_debug(
                    f"No more pairs meet minimum frequency threshold. Stopping at vocab size: {self.current_vocab_size}",
                    tag="N-GRAM BPE"
                )
                break

            # Assign new ID to this pair
            new_id = self.current_vocab_size
            frequency = pair_frequencies[most_frequent_pair]

            # Update progress bar with current vocab size
            pbar.set_postfix({"vocab_size": self.current_vocab_size, "freq": frequency})

            # Decode pair for human-readable output (only if verbose)
            if self.verbose and original_vocab:
                token1_id, token2_id = most_frequent_pair

                # Decode token1 (could be original or previously created n-gram)
                if token1_id < self.original_vocab_size:
                    token1_text = (
                        original_vocab[token1_id]
                        if token1_id < len(original_vocab)
                        else f"UNK_{token1_id}"
                    )
                else:
                    # This is a previously created n-gram, reconstruct it
                    token1_text = self.reconstruct_ngram_meaning(token1_id, original_vocab)

                # Decode token2 (could be original or previously created n-gram)
                if token2_id < self.original_vocab_size:
                    token2_text = (
                        original_vocab[token2_id]
                        if token2_id < len(original_vocab)
                        else f"UNK_{token2_id}"
                    )
                else:
                    # This is a previously created n-gram, reconstruct it
                    token2_text = self.reconstruct_ngram_meaning(token2_id, original_vocab)

                tqdm.write(
                    f"Iteration {iteration + 1}: Merging '{token1_text}'+'{token2_text}' "
                    f"(freq: {frequency}) -> ID {new_id}"
                )

            # Store merge operation with frequency
            self.merge_operations.append((most_frequent_pair, new_id, frequency))
            self.pair_to_id[most_frequent_pair] = new_id
            self.id_to_pair[new_id] = most_frequent_pair
            self.pair_frequencies[most_frequent_pair] = frequency

            # Apply merge to all documents
            working_data = self.merge_word_pairs(working_data, most_frequent_pair, new_id)

            # Update vocabulary size
            self.current_vocab_size += 1
            iteration += 1
            pbar.update(1)

            # Safety check to prevent infinite loops
            if iteration > 50000:
                pbar.close()
                self._console.print_warning("Maximum iterations reached. Stopping merge process.", tag="N-GRAM BPE")
                break

        pbar.close()
        self._console.print_debug(f"N-gram BPE completed. Final vocab size: {self.current_vocab_size}", tag="N-GRAM BPE")
        self._console.print_debug(f"Created {len(self.merge_operations)} n-gram combinations", tag="N-GRAM BPE")

        return working_data

    def get_ngram_vocab_info(self) -> Dict:
        """
        Return information about the n-gram vocabulary created.

        Returns:
            Dictionary containing vocabulary statistics and mappings
        """
        return {
            "original_vocab_size": self.original_vocab_size,
            "final_vocab_size": self.current_vocab_size,
            "ngrams_created": len(self.merge_operations),
            "merge_operations": self.merge_operations[:10],  # Show first 10 for inspection
            "pair_to_id_sample": dict(list(self.pair_to_id.items())[:5]),  # Sample mappings
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
        # Memoization: check cache first using vocab identity
        cache_key = (ngram_id, id(original_vocab))
        if cache_key in self._ngram_cache:
            return self._ngram_cache[cache_key]

        # Base cases
        if ngram_id < self.original_vocab_size:
            result = original_vocab[ngram_id] if ngram_id < len(original_vocab) else f"UNK_{ngram_id}"
        elif ngram_id not in self.id_to_pair:
            result = f"NGRAM_{ngram_id}"
        else:
            # Recursively reconstruct the pair
            word1_id, word2_id = self.id_to_pair[ngram_id]
            word1_str = self.reconstruct_ngram_meaning(word1_id, original_vocab)
            word2_str = self.reconstruct_ngram_meaning(word2_id, original_vocab)
            result = f"{word1_str}_{word2_str}"

        self._ngram_cache[cache_key] = result
        return result

    def save_ngrams_to_json(
        self, filename: str, original_vocab: List[str], output_dir: str = None
    ) -> str:
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

        # Helper to decode token ID to text
        def decode_token(token_id: int) -> str:
            if token_id < self.original_vocab_size:
                return original_vocab[token_id] if token_id < len(original_vocab) else f"UNK_{token_id}"
            return self.reconstruct_ngram_meaning(token_id, original_vocab)

        # Build ngrams dict using dict comprehension
        ngrams_data = {
            "metadata": {
                "original_vocab_size": self.original_vocab_size,
                "final_vocab_size": self.current_vocab_size,
                "ngrams_created": len(self.merge_operations),
                "vocab_limit": self.vocab_limit,
                "min_pair_frequency": self.min_pair_frequency,
            },
            "ngrams": {
                str(new_id): {
                    "pair": f"{decode_token(pair[0])},{decode_token(pair[1])}",
                    "frequency": self.pair_frequencies.get(pair, 0),
                }
                for pair, new_id in self.pair_to_id.items()
            },
        }

        # Save to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ngrams_data, f, ensure_ascii=False, indent=2)

        self._console.print_status(f"N-grams saved to: {filepath}", "success")
        return filepath
