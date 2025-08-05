"""Term Frequency (TF) weighting functions for TF-IDF calculations."""

import numpy as np
from scipy.sparse import csr_matrix

def tf_a(x: csr_matrix):
        t = x.copy()
        maximums = np.maximum.reduceat(t.data, t.indptr[:-1])
        row_counts = t.indptr[1:] - t.indptr[:-1]
        maximums = np.repeat(maximums, row_counts)
        t.data = 0.5 + 0.5 * t.data / maximums
        return t


def tf_b(x: csr_matrix):
    """Binary term frequency weighting."""
    t = x.copy()
    t.data = np.ones_like(x.data)
    return t


def tf_d(x: csr_matrix):
    """Double logarithm term frequency weighting."""
    t = x.copy()
    t.data = 1 + np.log2(1 + np.log2(t.data))
    return t


def tf_l(x: csr_matrix):
    """Logarithm term frequency weighting."""
    t = x.copy()
    t.data = 1 + np.log2(t.data)
    return t


def tf_L(x: csr_matrix):
    """
    Length-normalized logarithm term frequency weighting.
    
    This function calculates the TF-IDF score for a given matrix.
    Takes a csr_matrix and returns a csr_matrix.
    """
    t = x.copy()
    row_sums = np.add.reduceat(t.data, t.indptr[:-1])
    row_counts = t.indptr[1:] - t.indptr[:-1]
    row_avg = (1 + row_sums) / (1 + row_counts)
    payda = 1 + np.log2(row_avg)
    payda = np.repeat(payda, row_counts)
    pay = 1 + np.log2(t.data)
    t.data = pay / payda
    return t 