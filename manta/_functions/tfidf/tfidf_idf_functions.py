"""Inverse Document Frequency (IDF) weighting functions for TF-IDF calculations."""

import numpy as np


def idf_n(df: np.ndarray, document_count: int):
        return np.ones_like(df)

def idf_f(df: np.ndarray, document_count: int):
    return np.log2(document_count / df) + 1

def idf_t(df: np.ndarray, document_count: int):
    return np.log2((1 + document_count) / df)

def idf_p(df: np.ndarray, document_count: int):
    return np.log2((document_count - df + 1) / (df + 1))

def idf_bm25(df: np.ndarray, document_count: int):
    return np.log2((document_count - df + 0.5) / (df + 0.5))