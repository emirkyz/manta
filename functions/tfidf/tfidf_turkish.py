from collections import Counter

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tokenizers import Tokenizer

from .tf_funcs import *
from .idf_funcs import *


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