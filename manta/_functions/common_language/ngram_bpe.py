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

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import json
import os


class WordPairBPE:
    """
    BPE-like algorithm for creating n-gram word pairs from counterized data.

    This class implements a memory-efficient approach to generating meaningful
    word n-grams by iteratively merging the most frequent adjacent word pairs
    until a vocabulary size limit is reached.
    """

    def __init__(self, vocab_limit: int = 10000, min_pair_frequency: int = 2):
        """
        Initialize the WordPairBPE encoder.

        Args:
            vocab_limit: Maximum vocabulary size before stopping merging
            min_pair_frequency: Minimum frequency threshold for pair merging
        """
        self.vocab_limit = vocab_limit
        self.min_pair_frequency = min_pair_frequency
        self.original_vocab_size = 0
        self.current_vocab_size = 0
        self.merge_operations = []  # List of (pair, new_id, frequency) tuples
        self.pair_to_id = {}  # Mapping of merged pairs to their new IDs
        self.id_to_pair = {}  # Reverse mapping for reconstruction
        self.pair_frequencies = {}  # Store frequencies for each merged pair

    def build_pair_frequency_table(self, counterized_data: List[List[int]]) -> Counter:
        """
        Build frequency table of adjacent word pairs.

        Args:
            counterized_data: List of documents, each containing word IDs

        Returns:
            Counter object with (word1_id, word2_id) -> frequency mappings
        """
        pair_frequencies = Counter()

        for document in counterized_data:
            if len(document) < 2:
                continue

            # Count adjacent pairs in this document
            for i in range(len(document) - 1):
                pair = (document[i], document[i + 1])
                pair_frequencies[pair] += 1

        return pair_frequencies

    def find_most_frequent_pair(self, pair_frequencies: Counter) -> Optional[Tuple[int, int]]:
        """
        Find the most frequent word pair that meets minimum frequency threshold.

        Args:
            pair_frequencies: Counter with pair frequencies

        Returns:
            Most frequent pair as (word1_id, word2_id) tuple, or None if no valid pairs
        """
        if not pair_frequencies:
            return None

        most_frequent_pair, frequency = pair_frequencies.most_common(1)[0]

        if frequency >= self.min_pair_frequency:
            return most_frequent_pair
        else:
            return None

    def merge_word_pairs(self, counterized_data: List[List[int]],
                        pair_to_merge: Tuple[int, int], new_id: int) -> List[List[int]]:
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
                if (i < len(document) - 1 and
                    document[i] == word1 and document[i + 1] == word2):
                    # Replace pair with new ID
                    new_document.append(new_id)
                    i += 2  # Skip both words
                else:
                    # Keep current word as-is
                    new_document.append(document[i])
                    i += 1

            updated_data.append(new_document)

        return updated_data

    def fit(self, counterized_data: List[List[int]], original_vocab_size: int, original_vocab: List[str] = None) -> List[List[int]]:
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

        # Work with a copy to avoid modifying original data
        working_data = [doc[:] for doc in counterized_data]

        print(f"Starting n-gram BPE with vocab size: {self.current_vocab_size}")
        print(f"Target vocab limit: {self.vocab_limit}")

        iteration = 0
        while self.current_vocab_size < self.vocab_limit:
            # Build frequency table for current iteration
            pair_frequencies = self.build_pair_frequency_table(working_data)

            # Find most frequent pair
            most_frequent_pair = self.find_most_frequent_pair(pair_frequencies)

            if most_frequent_pair is None:
                print(f"No more pairs meet minimum frequency threshold. Stopping at vocab size: {self.current_vocab_size}")
                break

            # Assign new ID to this pair
            new_id = self.current_vocab_size
            frequency = pair_frequencies[most_frequent_pair]
            original_vocab = False
            # Decode pair for human-readable output
            if original_vocab:
                token1_id, token2_id = most_frequent_pair

                # Decode token1 (could be original or previously created n-gram)
                if token1_id < self.original_vocab_size:
                    token1_text = original_vocab[token1_id] if token1_id < len(original_vocab) else f"UNK_{token1_id}"
                else:
                    # This is a previously created n-gram, reconstruct it
                    token1_text = self.reconstruct_ngram_meaning(token1_id, original_vocab)

                # Decode token2 (could be original or previously created n-gram)
                if token2_id < self.original_vocab_size:
                    token2_text = original_vocab[token2_id] if token2_id < len(original_vocab) else f"UNK_{token2_id}"
                else:
                    # This is a previously created n-gram, reconstruct it
                    token2_text = self.reconstruct_ngram_meaning(token2_id, original_vocab)

                print(f"Iteration {iteration + 1}: Merging '{token1_text}'+'{token2_text}' "
                      f"(freq: {frequency}) -> ID {new_id}")

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

            # Safety check to prevent infinite loops
            if iteration > 50000:
                print("Warning: Maximum iterations reached. Stopping merge process.")
                break

        print(f"N-gram BPE completed. Final vocab size: {self.current_vocab_size}")
        print(f"Created {len(self.merge_operations)} n-gram combinations")

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
                "original_vocab_size": self.original_vocab_size,
                "final_vocab_size": self.current_vocab_size,
                "ngrams_created": len(self.merge_operations),
                "vocab_limit": self.vocab_limit,
                "min_pair_frequency": self.min_pair_frequency
            },
            "ngrams": {}
        }

        # Export each n-gram with its information
        for pair, new_id in self.pair_to_id.items():
            token1_id, token2_id = pair
            frequency = self.pair_frequencies.get(pair, 0)

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
                "frequency": frequency
            }

        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ngrams_data, f, ensure_ascii=False, indent=2)

        print(f"N-grams saved to: {filepath}")
        return filepath