from collections import Counter

import scipy
from scipy.sparse import lil_matrix

from .tfidf_tf_functions import *
from .tfidf_idf_functions import *
from .tfidf_bm25_turkish import bm25_generator


def tf_idf_english(N=None,
                   vocab=None,
                   data=None,
                   output_dir=None,
                   fieldname=None,
                   lemmatize=False,
                   use_bm25=False,
                   k1=1.2,
                   b=0.75,
                   use_pivoted_norm=True,
                   slope=0.2) -> scipy.sparse.csr.csr_matrix:
    """
    Calculates Term Frequency-Inverse Document Frequency (TF-IDF) or BM25 matrix from document collection.
    
    This function processes a collection of documents to create a TF-IDF or BM25 sparse matrix representation
    using modular TF and IDF weighting functions. It supports multiple weighting schemes including 
    logarithmic TF with length normalization, pivoted normalization, and BM25 scoring.
    
    Args:
        N (int, optional): Total number of documents in the dataset. Used for matrix dimension sizing.
        vocab (list, optional): Sorted vocabulary list where each word corresponds to a matrix column index.
                                The vocabulary should be pre-sorted for efficient binary search operations.
        data (pandas.DataFrame or dict, optional): Document collection containing the text data to process.
                                                  Should have a column/key specified by alanadi parameter.
        output_dir (str, optional): Directory path for saving output files. Currently unused in implementation.
        fieldname (str, optional): Field/column name in data containing the document texts to process.
        lemmatize (bool, optional): Whether to apply lemmatization during text preprocessing. Default is False.
                                  When True, reduces words to their base forms for better semantic grouping.
        use_bm25 (bool, optional): If True, use BM25 instead of TF-IDF (default: False).
        k1 (float, optional): BM25 term frequency saturation parameter (default: 1.2) [1.2-2.0].
        b (float, optional): BM25 length normalization parameter (default: 0.75) [0, 0.75, 1].
        use_pivoted_norm (bool, optional): Apply pivoted normalization to TF-IDF (default: True).
        slope (float, optional): Pivoted normalization slope parameter (default: 0.2) [0-1].
    
    Returns:
        scipy.sparse.csr_matrix: Sparse TF-IDF or BM25 matrix with shape (N, len(sozluk)) where:
                                - Rows represent documents
                                - Columns represent vocabulary terms
                                - Values are TF-IDF or BM25 scores
                                - Matrix uses CSR format for efficient arithmetic operations
    
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

    #update_progress_emit(50, "TF-IDF Hesaplanıyor", "PROCESSING", "tfidf", "tid")
    try:
        # Create initial term frequency matrix
        document_count = len(data)
        vocabulary_count = len(vocab)

        matris = lil_matrix((document_count, vocabulary_count), dtype=int)

        for i, document in enumerate(data):
            histogram = Counter(document)
            temporary = [(k, v) for k, v in histogram.items()]
            columns = [a[0] for a in temporary]
            values = [b[1] for b in temporary]
            matris[i, columns] = values

        matris = matris.tocsr()

        # Calculate document frequency (DF)
        input_matrix = matris.tocsc(copy=True)
        input_matrix.data = np.ones_like(input_matrix.data)
        df = np.add.reduceat(input_matrix.data, input_matrix.indptr[:-1])

        if use_bm25:
            # Use BM25 scoring
            tf_idf = bm25_generator(input_matrix, df, N, k1, b)
        else:
            # Use traditional TF-IDF scoring with modular functions
            idf = idf_t(df, N)
            tf_idf = tf_l(input_matrix).multiply(idf).tocsr()
            tf_idf.eliminate_zeros()
            
            # Apply pivoted normalization if enabled
            if use_pivoted_norm:
                # Calculate document lengths (number of terms in each document)
                doc_lengths = np.add.reduceat(matris.data, matris.indptr[:-1])
                avg_doc_length = np.mean(doc_lengths)

                if slope != -1:
                    # Apply pivoted normalization
                    # norm = (1 - slope) + slope * (doc_length / avg_doc_length)
                    pivoted_norms = (1 - slope) + slope * (doc_lengths / avg_doc_length)

                    # Normalize the term frequencies
                    # Repeat the normalization factors for each non-zero element in the row
                    nnz_per_row = np.diff(tf_idf.indptr)
                    tf_idf.data = tf_idf.data / np.repeat(pivoted_norms, nnz_per_row)

        # Print memory usage statistics
        required_memory = N * len(vocab) * 3 * 8 / 1024 / 1024 / 1024
        print("Required memory : ", required_memory, "GB")
        temp = tf_idf.tocoo()
        sparse_matrix_required_memory = temp.nnz * 3 * 8 / 1024 / 1024 / 1024
        method_name = "BM25" if use_bm25 else "TF-IDF"
        print(f"{method_name} required memory : ", sparse_matrix_required_memory, "GB")
        count_of_nonzero = tf_idf.count_nonzero()
        percentage_of_nonzero = count_of_nonzero / (N * len(vocab))
        print("Percentage of nonzero elements : ", percentage_of_nonzero)

        return tf_idf

    except Exception as e:
        print(f"Error: {e}")
        #update_progress_emit("100", e, "ABORTED", "tfidf", "tid")
        raise e 