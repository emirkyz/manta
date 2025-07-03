"""BM25 implementation for Turkish text processing."""

import numpy as np
from scipy.sparse import csr_matrix

from .tfidf_idf_functions import idf_bm25


def bm25_generator(matris: csr_matrix, df: np.ndarray, dokuman_sayisi: int, k1=1.2, b=0.75):
    """
    Generate BM25 scores for a given term-document matrix.
    
    BM25 formula: BM25(qi,D) = IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
    
    Args:
        matris (csr_matrix): Term-document frequency matrix
        df (np.ndarray): Document frequency for each term
        dokuman_sayisi (int): Total number of documents
        k1 (float): Term frequency saturation parameter (default: 1.2)
        b (float): Length normalization parameter (default: 0.75)
    
    Returns:
        csr_matrix: BM25 score matrix
    """
    
    # Calculate IDF using existing BM25 IDF function
    idf = idf_bm25(df, dokuman_sayisi)
    
    # Calculate document lengths (number of terms per document)
    doc_lengths = np.asarray(matris.sum(axis=1)).flatten()
    
    # Calculate average document length
    avgdl = np.mean(doc_lengths)
    
    # Create BM25 matrix
    bm25_matrix = matris.copy().astype(np.float64)
    
    # Apply BM25 transformation for each document
    for i in range(dokuman_sayisi):
        # Get document length for normalization
        doc_len = doc_lengths[i]
        
        # Get start and end indices for this document's data
        start_idx = bm25_matrix.indptr[i]
        end_idx = bm25_matrix.indptr[i + 1]
        
        if start_idx < end_idx:  # Document has terms
            # Get term frequencies for this document
            tf = bm25_matrix.data[start_idx:end_idx]
            
            # Get term indices for this document
            term_indices = bm25_matrix.indices[start_idx:end_idx]
            
            # Calculate BM25 scores
            # BM25 = IDF × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × doc_len/avgdl))
            normalization_factor = k1 * (1 - b + b * doc_len / avgdl)
            numerator = tf * (k1 + 1)
            denominator = tf + normalization_factor
            
            # Apply IDF weights
            bm25_scores = idf[term_indices] * (numerator / denominator)
            
            # Update the matrix data
            bm25_matrix.data[start_idx:end_idx] = bm25_scores
    
    # Eliminate zeros and return
    bm25_matrix.eliminate_zeros()
    
    return bm25_matrix