import bisect
from collections import Counter

import numpy as np
import scipy
from scipy.sparse import lil_matrix

from functions.english.english_preprocessor import preprocess
from utils.redis_bridge import update_progress_emit
from .tfidf_tf_functions import tf_L
from .tfidf_idf_functions import idf_t


def tfidf_hesapla(N=None, sozluk=None, data=None, output_dir=None, alanadi=None, lemmatize=False) -> scipy.sparse.csr.csr_matrix:
    """
    Calculates Term Frequency-Inverse Document Frequency (TF-IDF) matrix from document collection.
    
    This function processes a collection of documents to create a TF-IDF sparse matrix representation.
    It uses chunked processing to handle large datasets efficiently and provides memory usage statistics.
    The function supports optional lemmatization during text preprocessing and calculates optimal 
    topic modeling parameters based on the resulting matrix sparsity.
    
    Args:
        N (int, optional): Total number of documents in the dataset. Used for matrix dimension sizing.
        sozluk (list, optional): Sorted vocabulary list where each word corresponds to a matrix column index.
                                The vocabulary should be pre-sorted for efficient binary search operations.
        data (pandas.DataFrame or dict, optional): Document collection containing the text data to process.
                                                  Should have a column/key specified by alanadi parameter.
        output_dir (str, optional): Directory path for saving output files. Currently unused in implementation.
        alanadi (str, optional): Field/column name in data containing the document texts to process.
        lemmatize (bool, optional): Whether to apply lemmatization during text preprocessing. Default is False.
                                  When True, reduces words to their base forms for better semantic grouping.
    
    Returns:
        scipy.sparse.lil_matrix: Sparse TF-IDF matrix with shape (N, len(sozluk)) where:
                                - Rows represent documents
                                - Columns represent vocabulary terms
                                - Values are TF-IDF scores
                                - Matrix uses LIL (List of Lists) format for efficient incremental construction
    
    Side Effects:
        - Prints lemmatization status at function start
        - Prints detailed memory usage statistics including:
          * Required memory for dense matrix representation
          * Actual memory usage of sparse matrix
          * Count of non-zero elements
          * Optimal topic count estimation
          * Sparsity percentage
        - Updates progress via Redis bridge (currently commented out)
        - May raise and re-propagate exceptions with progress updates
    """
    if lemmatize: 
        print("Lemmatization is enabled")
    else: 
        print("Lemmatization is disabled")

    chunk_size = 1000
    #update_progress_emit(50, "TF-IDF Hesaplanıyor", "PROCESSING", "tfidf", "tid")
    try:
        start_progress = 50
        end_progress = 70
        tfidf = lil_matrix((N, len(sozluk)))

        # Process documents in chunks
        for chunk_start in range(0, N, chunk_size):
            # Calculate the end of the current chunk (making sure not to exceed N)
            chunk_end = min(chunk_start + chunk_size, N)

            # Process each document in the current chunk
            for i in range(chunk_start, chunk_end):
                # Update progress for each document, just like in the original code
                dokuman = data[alanadi][i]
                metin_parcalari = preprocess(dokuman, lemmatize=lemmatize, kategoriler=frozenset(['Ll', 'Zs']))
                sozluk_sira_nolari = [bisect.bisect_left(sozluk, kelime) for kelime in metin_parcalari]
                frekanslar = Counter(sozluk_sira_nolari)

                sutun_sira_nolari = np.array(list(frekanslar.keys()))
                frekans_degerleri = np.array(list(frekanslar.values()))

                tfidf[i, sutun_sira_nolari] = frekans_degerleri
                #if i % 100 == 0:
                #    update_progress_emit(progress=int(start_progress + (end_progress - start_progress) * i // N),
                #                         message=f"Processing document {i + 1}/{N}",
                #                         status="PROCESSING",
                #                         data_name="tfidf",
                #                         tid="tid")

        gercek_gerekli_alan = N * len(sozluk) * 3 * 8 / 1024 / 1024 / 1024
        print("Gerekli alan : ", gercek_gerekli_alan, "GB")
        temp = tfidf.tocoo()
        seyrek_matris_gerekli_alan = temp.nnz * 3 * 8 / 1024 / 1024 / 1024
        print("tf-idf gerekli alan : ", seyrek_matris_gerekli_alan, "GB")
        counnt_of_nonzero = tfidf.count_nonzero()
        print("tf-idf count nonzero : ", counnt_of_nonzero)
        max_optimal_topic_num = counnt_of_nonzero // (N + len(sozluk))
        print("max_optimal_topic_num : ", max_optimal_topic_num)
        percentage_of_nonzero = counnt_of_nonzero / (N * len(sozluk))
        print("percentage_of_nonzero : ", percentage_of_nonzero)
        return tfidf

    except Exception as e:
        print(f"Error: {e}")
        #update_progress_emit("100", e, "ABORTED", "tfidf", "tid")
        raise e 