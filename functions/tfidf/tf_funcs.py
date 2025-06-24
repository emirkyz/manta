"""Term Frequency (TF) weighting functions for TF-IDF calculations."""

import numpy as np
from scipy.sparse import csr_matrix

def tf_a(x: csr_matrix):
    """
    Apply augmented term frequency weighting using 0.5 + 0.5 * (tf/max_tf).
    
    Args:
        x (csr_matrix): Sparse matrix containing term frequencies
    
    Returns:
        csr_matrix: Augmented term frequency matrix
    """
    t = x.copy()
    maximumlar = np.maximum.reduceat(t.data, t.indptr[:-1])
    eleman_sayilari = t.indptr[1:] - t.indptr[:-1]
    maximumlar = np.repeat(maximumlar, eleman_sayilari)
    t.data = 0.5 + 0.5 * t.data / maximumlar
    return t


def tf_b(x: csr_matrix):
    """
    Apply binary term frequency weighting (presence/absence of terms).
    
    Args:
        x (csr_matrix): Sparse matrix containing term frequencies
    
    Returns:
        csr_matrix: Binary term frequency matrix where all non-zero values become 1
    """
    t = x.copy()
    t.data = np.ones_like(x.data)
    return t


def tf_d(x: csr_matrix):
    """
    Apply double logarithm term frequency weighting using 1 + log2(1 + log2(tf)).
    
    Args:
        x (csr_matrix): Sparse matrix containing term frequencies
    
    Returns:
        csr_matrix: Double logarithm weighted term frequency matrix
    """
    t = x.copy()
    t.data = 1 + np.log2(1 + np.log2(t.data))
    return t


def tf_l(x: csr_matrix):
    """
    Apply logarithm term frequency weighting using 1 + log2(tf).
    
    Args:
        x (csr_matrix): Sparse matrix containing term frequencies
    
    Returns:
        csr_matrix: Logarithm weighted term frequency matrix
    """
    t = x.copy()
    t.data = 1 + np.log2(t.data)
    return t


def tf_L(x: csr_matrix):
    """
    Apply length-normalized logarithm term frequency weighting.
    
    Uses (1 + log2(tf)) / (1 + log2(avg_tf_per_doc)) for normalization.
    
    Args:
        x (csr_matrix): Sparse matrix containing term frequencies
    
    Returns:
        csr_matrix: Length-normalized logarithm weighted term frequency matrix
    """
    t = x.copy()
    satir_toplamlari = np.add.reduceat(t.data, t.indptr[:-1])
    eleman_sayilari = t.indptr[1:] - t.indptr[:-1]
    satir_ortalama = (1 + satir_toplamlari) / (1 + eleman_sayilari)
    payda = 1 + np.log2(satir_ortalama)
    payda = np.repeat(payda, eleman_sayilari)
    pay = 1 + np.log2(t.data)
    t.data = pay / payda
    return t 