from collections import Counter

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tokenizers import Tokenizer


def tf_b(x: csr_matrix):
    t = x.copy()
    t.data = np.ones_like(x.data)
    return t


def tf_d(x: csr_matrix):
    t = x.copy()
    t.data = 1 + np.log2(1 + np.log2(t.data))
    return t


def tf_l(x: csr_matrix):
    t = x.copy()
    t.data = 1 + np.log2(t.data)
    return t


def tf_L(x: csr_matrix):
    """
    This function calculates the TF-IDF score for a given matrix.
    Takes a csr_matrix and returns a csr_matrix.
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


def idf_t(df: np.ndarray, dokuman_sayisi: int):
    """
    This function calculates the IDF score for a given array.
    Takes an array and the number of documents and returns an array.
    """
    return np.log2((1 + dokuman_sayisi) / df)

def idf_p(df: np.ndarray, dokuman_sayisi: int):
    return np.log2((dokuman_sayisi - df + 1) / (df + 1))


def tf_idf_generator(veri, tokenizer: Tokenizer):
    """
    This function generates a TF-IDF matrix for a given list of text data.
    1) Convert the text data to a sparse matrix.
    2) Calculate the TF-IDF score for the sparse matrix.
    3) Return the TF-IDF matrix.
    
    Args:
        veri (list): A list of text data.
        tokenizer (Tokenizer): A trained tokenizer.

    Returns:
        csr_matrix: A sparse TF-IDF matrix.
    """
    
    dokuman_sayisi = len(veri)
    kelime_sayisi = tokenizer.get_vocab_size()

    matris = lil_matrix((dokuman_sayisi, kelime_sayisi), dtype=int)

    for i, dokuman in enumerate(veri):
        histogram = Counter(dokuman)
        gecici = [(k, v) for k, v in histogram.items()]
        sutunlar = [a[0] for a in gecici]
        degerler = [b[1] for b in gecici]
        matris[i, sutunlar] = degerler

    matris = matris.tocsr()

    df_girdi_matrisi = matris.tocsc(copy=True)
    df_girdi_matrisi.data = np.ones_like(df_girdi_matrisi.data)
    df = np.add.reduceat(df_girdi_matrisi.data, df_girdi_matrisi.indptr[:-1])
    idf = idf_p(df, dokuman_sayisi)

    tf_idf = tf_L(matris).multiply(idf).tocsr()
    tf_idf.eliminate_zeros()

    sozluk = list(tokenizer.get_vocab().keys())
    N = len(veri)

    gercek_gerekli_alan = N * len(sozluk) * 3 * 8 / 1024 / 1024 / 1024
    print("Gerekli alan : ", gercek_gerekli_alan, "GB")
    temp = tf_idf.tocoo()
    seyrek_matris_gerekli_alan = temp.nnz * 3 * 8 / 1024 / 1024 / 1024
    print("tf-idf gerekli alan : ", seyrek_matris_gerekli_alan, "GB")
    counnt_of_nonzero = tf_idf.count_nonzero()
    print("tf-idf count nonzero : ", counnt_of_nonzero)
    total_elements = tf_idf.shape[0] * tf_idf.shape[1]
    print("tf-idf total elements : ", total_elements)
    max_optimal_topic_num = counnt_of_nonzero // (N + len(sozluk))
    print("max_optimal_topic_num : ", max_optimal_topic_num)

    return tf_idf
