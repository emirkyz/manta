"""Inverse Document Frequency (IDF) weighting functions for TF-IDF calculations."""

import numpy as np


def idf_n(df: np.ndarray, dokuman_sayisi: int):
        return np.ones_like(df)

def idf_f(df: np.ndarray, dokuman_sayisi: int):
    return np.log2(dokuman_sayisi / df)

def idf_t(df: np.ndarray, dokuman_sayisi: int):
    return np.log2((1 + dokuman_sayisi) / df)

def idf_p(df: np.ndarray, dokuman_sayisi: int):
    return np.log2((dokuman_sayisi - df + 1) / (df + 1))

