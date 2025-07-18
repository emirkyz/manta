from collections import Counter

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tokenizers import Tokenizer

from .tfidf_tf_functions import *
from .tfidf_idf_functions import *
from .tfidf_bm25_turkish import bm25_generator


def tf_idf_turkish(veri, tokenizer: Tokenizer, use_bm25=False, k1=1.2, b=0.75):
    """
    This function generates a TF-IDF or BM25 matrix for a given list of text data.
    1) Convert the text data to a sparse matrix.
    2) Calculate the TF-IDF or BM25 score for the sparse matrix.
    3) Return the TF-IDF or BM25 matrix.
    
    Args:
        veri (list): A list of text data.
        tokenizer (Tokenizer): A trained tokenizer.
        use_bm25 (bool): If True, use BM25 instead of TF-IDF (default: False).
        k1 (float): BM25 term frequency saturation parameter (default: 1.2) [1.2-2.0].
        b (float): BM25 length normalization parameter (default: 0.75)[0, 0.75, 1].

    Returns:
        csr_matrix: A sparse TF-IDF or BM25 matrix.
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
    #df = np.array((df_girdi_matrisi > 0).sum(axis=0)).flatten()
    df = np.add.reduceat(df_girdi_matrisi.data, df_girdi_matrisi.indptr[:-1])

    use_bm25 = False
    if use_bm25:
        # Use BM25 scoring
        tf_idf = bm25_generator(matris, df, dokuman_sayisi, k1, b)
    else:
        # Use traditional TF-IDF scoring
        idf = idf_p(df, dokuman_sayisi)
        tf_idf = tf_L(matris).multiply(idf).tocsr()
        tf_idf.eliminate_zeros()
        
        # Calculate document lengths for pivoted normalization
    use_pivoted_norm = True
    slope = 0.2
    if use_pivoted_norm and not use_bm25:
        # Calculate document lengths (number of terms in each document)
        doc_lengths = np.add.reduceat(matris.data, matris.indptr[:-1])
        avg_doc_length = np.mean(doc_lengths)
        
        # Apply pivoted normalization
        # norm = (1 - slope) + slope * (doc_length / avg_doc_length)
        pivoted_norms = (1 - slope) + slope * (doc_lengths / avg_doc_length)
        
        # Normalize the term frequencies
        # Repeat the normalization factors for each non-zero element in the row
        nnz_per_row = np.diff(matris.indptr)
        tf_idf.data = tf_idf.data / np.repeat(pivoted_norms, nnz_per_row)


    '''    
    norm_rows = np.sqrt(np.add.reduceat(np.log(tf_idf.data) * np.log(tf_idf.data), tf_idf.indptr[:-1]))
    nnz_per_row = np.diff(tf_idf.indptr)
    tf_idf.data /= np.repeat(norm_rows, nnz_per_row)
    '''
    sozluk = list(tokenizer.get_vocab().keys())
    N = len(veri)

    gercek_gerekli_alan = N * len(sozluk) * 3 * 8 / 1024 / 1024 / 1024
    print("Gerekli alan : ", gercek_gerekli_alan, "GB")
    temp = tf_idf.tocoo()
    seyrek_matris_gerekli_alan = temp.nnz * 3 * 8 / 1024 / 1024 / 1024
    method_name = "BM25" if use_bm25 else "TF-IDF"
    print(f"{method_name} gerekli alan : ", seyrek_matris_gerekli_alan, "GB")
    counnt_of_nonzero = tf_idf.count_nonzero()
    print(f"{method_name} count nonzero : ", counnt_of_nonzero)
    total_elements = tf_idf.shape[0] * tf_idf.shape[1]
    print(f"{method_name} total elements : ", total_elements)
    max_optimal_topic_num = counnt_of_nonzero // (N + len(sozluk))
    print("max_optimal_topic_num : ", max_optimal_topic_num)

    return tf_idf 