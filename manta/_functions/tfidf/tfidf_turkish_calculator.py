from collections import Counter

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
    
    document_counts = len(veri)
    word_count = tokenizer.get_vocab_size()

    matris = lil_matrix((document_counts, word_count), dtype=int)

    for i, document in enumerate(veri):
        histogram = Counter(document)
        temporary = [(k, v) for k, v in histogram.items()]
        columns = [a[0] for a in temporary]
        values = [b[1] for b in temporary]
        matris[i, columns] = values

    matris = matris.tocsr()

    input_matrix = matris.tocsc(copy=True)
    input_matrix.data = np.ones_like(input_matrix.data)
    #df = np.array((df_input_matrix > 0).sum(axis=0)).flatten()
    df = np.add.reduceat(input_matrix.data, input_matrix.indptr[:-1])

    use_bm25 = False
    if use_bm25:
        # Use BM25 scoring
        tf_idf = bm25_generator(input_matrix, df, document_counts, k1, b)
    else:
        # Use traditional TF-IDF scoring
        idf = idf_p(df, document_counts)
        tf_idf = tf_L(input_matrix).multiply(idf).tocsr()
        tf_idf.eliminate_zeros()
        
        # Calculate document lengths for pivoted normalization
    use_pivoted_norm = True
    slope = 0.2
    if use_pivoted_norm and not use_bm25:
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


    '''    
    norm_rows = np.sqrt(np.add.reduceat(np.log(tf_idf.data) * np.log(tf_idf.data), tf_idf.indptr[:-1]))
    nnz_per_row = np.diff(tf_idf.indptr)
    tf_idf.data /= np.repeat(norm_rows, nnz_per_row)
    '''
    vocab = list(tokenizer.get_vocab().keys())
    N = len(veri)
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