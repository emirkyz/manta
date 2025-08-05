"""
Memory-efficient word co-occurrence frequency analysis using sparse matrices.

This module provides functionality to analyze word co-occurrence patterns in text
using sliding window approach with memory-efficient sparse matrix operations.
"""

import os
import json
import gc
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
from tokenizers import Tokenizer

from .._functions.turkish.turkish_preprocessor import process_text, clean_text_turkish
from .._functions.common_language.emoji_processor import EmojiMap


class WordCooccurrenceAnalyzer:
    """
    Memory-efficient word co-occurrence analyzer using sparse matrices.
    
    This class provides methods to analyze word co-occurrence patterns in text
    using sliding window approach optimized for memory efficiency with large datasets.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        min_count: int = 2,
        max_vocab_size: Optional[int] = None,
        language: str = "turkish",
        batch_size: int = 1000
    ):
        """
        Initialize the word co-occurrence analyzer.
        
        Args:
            window_size: Size of the sliding window for co-occurrence calculation
            min_count: Minimum frequency threshold for vocabulary filtering
            max_vocab_size: Maximum vocabulary size (keeps most frequent words)
            language: Language for text processing ("turkish" or "english")
            batch_size: Number of documents to process in each batch
        """
        self.window_size = window_size
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.language = language
        self.batch_size = batch_size
        
        # Initialize vocabulary and co-occurrence matrix
        self.vocabulary: Dict[str, int] = {}
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.cooccurrence_matrix: Optional[csr_matrix] = None
        
        # Initialize text processor
        self.emoji_processor = EmojiMap() if language == "turkish" else None
        
    def build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from input texts with frequency filtering.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count word frequencies
        word_counts = Counter()
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing text {i}/{len(texts)}")
            
            # Process text based on language
            if self.language == "turkish":
                processed_text = process_text(text, self.emoji_processor)
            else:
                processed_text = text.lower()
            
            # Tokenize and count words
            tokens = processed_text.split()
            word_counts.update(tokens)
        
        # Filter vocabulary by minimum count
        filtered_vocab = {word: count for word, count in word_counts.items() 
                         if count >= self.min_count}
        
        # Limit vocabulary size if specified
        if self.max_vocab_size and len(filtered_vocab) > self.max_vocab_size:
            filtered_vocab = dict(word_counts.most_common(self.max_vocab_size))
        
        # Create word-to-id mappings
        self.word_counts = Counter(filtered_vocab)
        self.word_to_id = {word: idx for idx, word in enumerate(filtered_vocab.keys())}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        vocab_size = len(self.word_to_id)
        print(f"Vocabulary built: {vocab_size} unique words")
        
        return filtered_vocab
    
    def _process_text_batch(self, texts: List[str]) -> List[Tuple[int, int, int]]:
        """
        Process a batch of texts and extract co-occurrence pairs.
        
        Args:
            texts: Batch of texts to process
            
        Returns:
            List of (word1_id, word2_id, count) tuples
        """
        cooccurrence_pairs = defaultdict(int)
        
        for text in texts:
            # Process text based on language
            if self.language == "turkish":
                processed_text = process_text(text, self.emoji_processor)
            else:
                processed_text = text.lower()
            
            # Tokenize
            tokens = processed_text.split()
            
            # Filter tokens to vocabulary
            valid_tokens = [token for token in tokens if token in self.word_to_id]
            
            # Extract co-occurrence pairs using sliding window
            for i, word1 in enumerate(valid_tokens):
                word1_id = self.word_to_id[word1]
                
                # Look at words within window
                start = max(0, i - self.window_size)
                end = min(len(valid_tokens), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        word2 = valid_tokens[j]
                        word2_id = self.word_to_id[word2]
                        
                        # Use lexicographic ordering to avoid duplicates
                        if word1_id < word2_id:
                            cooccurrence_pairs[(word1_id, word2_id)] += 1
                        elif word1_id > word2_id:
                            cooccurrence_pairs[(word2_id, word1_id)] += 1
        
        # Convert to list of tuples
        return [(i, j, count) for (i, j), count in cooccurrence_pairs.items()]
    
    def build_cooccurrence_matrix(self, texts: List[str]) -> csr_matrix:
        """
        Build co-occurrence matrix from texts using memory-efficient batch processing.
        
        Args:
            texts: List of input texts
            
        Returns:
            Sparse co-occurrence matrix
        """
        print(f"Building co-occurrence matrix from {len(texts)} texts...")
        
        # Build vocabulary first
        if not self.word_to_id:
            self.build_vocabulary(texts)
        
        vocab_size = len(self.word_to_id)
        
        # Process texts in batches to manage memory
        all_pairs = []
        
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//self.batch_size + 1}/"
                  f"{(len(texts) + self.batch_size - 1) // self.batch_size}")
            
            # Process batch and collect co-occurrence pairs
            batch_pairs = self._process_text_batch(batch_texts)
            all_pairs.extend(batch_pairs)
            
            # Force garbage collection to free memory
            gc.collect()
        
        # Build sparse matrix from collected pairs
        if all_pairs:
            rows, cols, data = zip(*all_pairs)
            
            # Create COO matrix and convert to CSR for efficiency
            cooccurrence_coo = coo_matrix((data, (rows, cols)), 
                                        shape=(vocab_size, vocab_size))
            
            # Make matrix symmetric
            cooccurrence_coo = cooccurrence_coo + cooccurrence_coo.T
            
            # Convert to CSR for efficient operations
            self.cooccurrence_matrix = cooccurrence_coo.tocsr()
        else:
            # Create empty matrix if no pairs found
            self.cooccurrence_matrix = csr_matrix((vocab_size, vocab_size))
        
        print(f"Co-occurrence matrix built: {vocab_size}x{vocab_size} "
              f"with {self.cooccurrence_matrix.nnz} non-zero entries")
        
        return self.cooccurrence_matrix
    
    def get_top_cooccurrences(self, top_n: int = 100) -> List[Tuple[str, str, float]]:
        """
        Get top co-occurring word pairs.
        
        Args:
            top_n: Number of top pairs to return
            
        Returns:
            List of (word1, word2, score) tuples sorted by score
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("Co-occurrence matrix not built yet")
        
        # Get upper triangular part to avoid duplicates
        upper_triangle = sparse.triu(self.cooccurrence_matrix, k=1)
        
        # Find non-zero entries and their values
        rows, cols = upper_triangle.nonzero()
        values = upper_triangle.data
        
        # Sort by values in descending order
        sorted_indices = np.argsort(values)[::-1]
        
        # Get top pairs
        top_pairs = []
        for idx in sorted_indices[:top_n]:
            row, col = rows[idx], cols[idx]
            value = values[idx]
            word1 = self.id_to_word[row]
            word2 = self.id_to_word[col]
            top_pairs.append((word1, word2, float(value)))
        
        return top_pairs
    
    def save_results(
        self,
        output_dir: str,
        table_name: str,
        top_n: int = 100,
        create_heatmap: bool = True,
        heatmap_size: int = 20,
        create_output_folder: bool = True
    ) -> Dict[str, Any]:
        """
        Save co-occurrence results to JSON and create visualizations.
        
        Args:
            output_dir: Directory to save results
            table_name: Name for output files
            top_n: Number of top pairs to save
            create_heatmap: Whether to create heatmap visualization
            heatmap_size: Number of words to include in heatmap
            
        Returns:
            Dictionary containing the results
        """
        # Create output directory
        if create_output_folder:
            output_path = Path(output_dir) / "Output" / table_name
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get top co-occurrences
        top_pairs = self.get_top_cooccurrences(top_n)
        
        # Prepare data for JSON
        results = {
            "pairs": [
                {"word_1": pair[0], "word_2": pair[1], "score": pair[2]}
                for pair in top_pairs
            ],
            "vocabulary_size": len(self.word_to_id),
            "total_cooccurrences": int(self.cooccurrence_matrix.nnz),
            "window_size": self.window_size,
            "min_count": self.min_count,
            "language": self.language
        }
        
        # Save to JSON
        json_file = output_path / f"{table_name}_cooccurrence.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Results saved to: {json_file}")
        
        # Create heatmap if requested
        if create_heatmap and top_pairs:
            self._create_heatmap(top_pairs, output_path, table_name, heatmap_size)
        
        return results
    
    def _create_heatmap(
        self,
        top_pairs: List[Tuple[str, str, float]],
        output_path: Path,
        table_name: str,
        heatmap_size: int
    ) -> None:
        """
        Create and save heatmap visualization.
        
        Args:
            top_pairs: List of top co-occurring pairs
            output_path: Path to save heatmap
            table_name: Name for the heatmap file
            heatmap_size: Number of words to include in heatmap
        """
        print("Creating co-occurrence heatmap...")
        
        # Get top words by total co-occurrence scores
        word_scores = defaultdict(float)
        for word1, word2, score in top_pairs:
            word_scores[word1] += score
            word_scores[word2] += score
        
        # Get top words for heatmap
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:heatmap_size]
        selected_words = [word for word, _ in top_words]
        
        # Create word-to-index mapping for heatmap
        word_to_idx = {word: idx for idx, word in enumerate(selected_words)}
        
        # Create heatmap matrix
        heatmap_matrix = np.zeros((len(selected_words), len(selected_words)))
        
        # Fill matrix with co-occurrence scores
        for word1, word2, score in top_pairs:
            if word1 in word_to_idx and word2 in word_to_idx:
                i, j = word_to_idx[word1], word_to_idx[word2]
                heatmap_matrix[i, j] = score
                heatmap_matrix[j, i] = score
        
        # Create heatmap
        plt.figure(figsize=(max(12, len(selected_words) * 0.8), 
                          max(10, len(selected_words) * 0.8)))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))
        
        # Determine annotation format
        max_score = np.max(heatmap_matrix)
        if max_score < 1:
            fmt = '.3f'
        elif max_score < 10:
            fmt = '.2f'
        else:
            fmt = '.1f'
        
        # Create heatmap
        sns.heatmap(
            heatmap_matrix,
            mask=mask,
            xticklabels=selected_words,
            yticklabels=selected_words,
            annot=True,
            fmt=fmt,
            cmap='YlOrRd',
            cbar_kws={'label': 'Co-occurrence Score'},
            square=True,
            linewidths=0.5,
            annot_kws={'size': max(8, 12 - len(selected_words) // 3)}
        )
        
        plt.title(f'Word Co-occurrence Heatmap - {table_name}', fontsize=16, pad=20)
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save heatmap
        heatmap_file = output_path / f"{table_name}_cooccurrence_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to: {heatmap_file}")


def analyze_word_cooccurrence(
    input_data: Union[str, List[str], pd.DataFrame],
    text_column: Optional[str] = None,
    window_size: int = 5,
    min_count: int = 2,
    max_vocab_size: Optional[int] = None,
    output_dir: str = "./",
    table_name: str = "cooccurrence_analysis",
    language: str = "turkish",
    create_heatmap: bool = True,
    heatmap_size: int = 20,
    top_n: int = 100,
    batch_size: int = 1000,
    create_output_folder: bool = True
) -> Dict[str, Any]:
    """
    Analyze word co-occurrence frequencies using memory-efficient sparse matrices.
    
    Args:
        input_data: Input data (file path, list of texts, or DataFrame)
        text_column: Column name if input_data is DataFrame
        window_size: Size of sliding window for co-occurrence
        min_count: Minimum word frequency threshold
        max_vocab_size: Maximum vocabulary size
        output_dir: Directory to save results
        table_name: Name for output files
        language: Language for text processing
        create_heatmap: Whether to create heatmap visualization
        heatmap_size: Number of words in heatmap
        top_n: Number of top pairs to return
        batch_size: Batch size for processing
        
    Returns:
        Dictionary containing analysis results
    """
    # Process input data
    if isinstance(input_data, str):
        # Assume it's a file path
        if input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
            if text_column is None:
                text_column = df.columns[0]
            texts = df[text_column].dropna().astype(str).tolist()
        else:
            # Assume it's a single text
            texts = [input_data]
    elif isinstance(input_data, list):
        texts = input_data
    elif isinstance(input_data, pd.DataFrame):
        if text_column is None:
            text_column = input_data.columns[0]
        texts = input_data[text_column].dropna().astype(str).tolist()
    else:
        raise ValueError("Unsupported input data type")
    
    # Initialize analyzer
    analyzer = WordCooccurrenceAnalyzer(
        window_size=window_size,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
        language=language,
        batch_size=batch_size
    )
    
    # Build co-occurrence matrix
    analyzer.build_cooccurrence_matrix(texts)
    
    # Save results
    results = analyzer.save_results(
        output_dir=output_dir,
        table_name=table_name,
        top_n=top_n,
        create_heatmap=create_heatmap,
        heatmap_size=heatmap_size,
        create_output_folder=create_output_folder
    )
    
    return results