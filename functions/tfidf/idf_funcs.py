"""Inverse Document Frequency (IDF) weighting functions for TF-IDF calculations."""

import numpy as np


def idf_n(df: np.ndarray, dokuman_sayisi: int):
    """
    Apply no IDF weighting (all weights are 1).
    
    Args:
        df (np.ndarray): Document frequency array
        dokuman_sayisi (int): Total number of documents
    
    Returns:
        np.ndarray: Array of ones with same shape as df
    """
    return np.ones_like(df)

def idf_f(df: np.ndarray, dokuman_sayisi: int):
    """
    Calculate standard IDF scores using log2(N/df).
    
    Args:
        df (np.ndarray): Document frequency array
        dokuman_sayisi (int): Total number of documents
    
    Returns:
        np.ndarray: Standard IDF scores
    """
    return np.log2(dokuman_sayisi / df)

def idf_t(df: np.ndarray, dokuman_sayisi: int):
    """
    Calculate smoothed IDF scores using log2((1+N)/df).
    
    Args:
        df (np.ndarray): Document frequency array
        dokuman_sayisi (int): Total number of documents
    
    Returns:
        np.ndarray: Smoothed IDF scores (avoids division by zero)
    """
    return np.log2((1 + dokuman_sayisi) / df)

def idf_p(df: np.ndarray, dokuman_sayisi: int):
    """
    Calculate probabilistic IDF scores using log2((N-df+1)/(df+1)).
    
    Args:
        df (np.ndarray): Document frequency array
        dokuman_sayisi (int): Total number of documents
    
    Returns:
        np.ndarray: Probabilistic IDF scores
    """
    return np.log2((dokuman_sayisi - df + 1) / (df + 1))

