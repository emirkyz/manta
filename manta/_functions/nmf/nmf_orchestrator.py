import scipy
from datetime import datetime
from typing import Callable
import numpy as np
from scipy import sparse as sp

from .nmf_initialization import nmf_initialization_nndsvd, nmf_initialization_random
from .nmf_projective_basic import projective_nmf
from .nmf_basic import _basic_nmf
from .nmtf.nmtf import nmtf
from .nmtf.nmtf_init import nmtf_initialization_random

def find_max_rank(sparse_matrix: sp.csc_matrix) -> int:
    """
    Computes the theoretical maximum rank of a given sparse matrix.

    The theoretical maximum rank of a matrix is determined by the smaller of its two dimensions:
    the number of rows and the number of columns. This function takes a sparse matrix as input
    and returns its maximum possible rank.

    :param sparse_matrix: Input sparse matrix in CSC (Compressed Sparse Column) format.
    :type sparse_matrix: sp.csc_matrix
    :return: The theoretical maximum rank of the input matrix.
    :rtype: int
    """
    if not sp.isspmatrix_csc(sparse_matrix):
        raise ValueError("Input matrix must be in CSC (Compressed Sparse Column) format.")

    # max_rank <= nnz(V) / (m + n)

    # count of non-zero elements
    counnt_of_nonzero = sparse_matrix.count_nonzero()
    print(f"Count of nonzero elements : ", counnt_of_nonzero)
    N = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    print(f"Count of total elements : ", N)

    # m = number of documents
    # n = number of words
    m, n = sparse_matrix.shape
    max_rank = counnt_of_nonzero / (m + n)
    print(f"Max theoretical rank : ", max_rank)

    max_rank = int(max_rank)
    return max_rank



def _nmf_cpu(in_mat: sp.csc_matrix, log: bool = True, rank_factor: float = 1.0,
             norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
             init_func: Callable = nmf_initialization_nndsvd,
             topic_count=-1,
             nmf_method:str = "nmf") -> dict[sp.csr_matrix, sp.csc_matrix]:
    """
    Performs Non-Negative Matrix Factorization (NMF) on the input sparse matrix
    while offering customization options for initialization, logging, thresholding, and
    different NMF methods.

    This function processes a compressed sparse matrix (CSC format) to produce two
    matrices, W and H, which approximate the input matrix such that:
        input_matrix ≈ W @ H

    Sections of the computation include initialization, the choice of NMF method,
    and thresholding for convergence. The NMF can optionally log progress during
    execution. Different NMF methods, such as basic or projective NMF, can be
    selected depending on the use case.

    :param in_mat: Input sparse matrix, expected to be in CSC (Compressed Sparse
        Column) format.
    :type in_mat: sp.csc_matrix
    :param log: Boolean flag to enable or disable logging of progress during
        execution. Defaults to True.
    :type log: bool
    :param rank_factor: Factor that influences the rank of the computed factorization.
        Defaults to 1.0.
    :type rank_factor: float
    :param norm_thresh: Convergence threshold for column normalization. If the change
        in column norm for two consecutive iterations does not exceed this value,
        convergence is assumed. Defaults to 1.0.
    :type norm_thresh: float
    :param zero_threshold: Minimum value threshold for matrix entries. Entries below
        this value are treated as zero. Defaults to 0.0001.
    :type zero_threshold: float
    :param init_func: Initialization function for the NMF process. This function
        determines the initial values of the W and H matrices. Defaults to
        ``nmf_initialization_nndsvd``.
    :type init_func: Callable
    :param konu_sayisi: Number of topics or latent factors to generate. Defaults to -1.
    :type konu_sayisi: int
    :param nmf_method: Specifies the NMF variant to use. Options include "basic"
        for basic NMF and "opnmf" for projective NMF. Defaults to "nmf".
    :type nmf_method: str
    :return: A tuple with two sparse matrices:
        - W: The left factor matrix in CSR (Compressed Sparse Row) format.
        - H: The right factor matrix in CSC (Compressed Sparse Column) format.
        Together, these matrices approximate the input matrix.
    :rtype: tuple[sp.csr_matrix, sp.csc_matrix]
    """

    # Find theorotical max rank of the input matrix

    if topic_count == -1:
        topic_count = find_max_rank(in_mat)


    w, h = init_func(in_mat, topic_count)

    if log:
        print("Starting Factorization Process..")
        start = datetime.now()

    nmf_output = None

    if nmf_method == "pnmf":
        # If projective NMF is used, we do not need to run the core NMF function
        nmf_output = projective_nmf(in_mat, r=topic_count, W_mat=w,h_mat=h,norm_func=np.linalg.norm)
    elif nmf_method == "nmf":
        nmf_output = _basic_nmf(in_mat, w, h, start, log=log, norm_thresh=norm_thresh, zero_threshold=zero_threshold,
                       norm_func=np.linalg.norm)
    elif nmf_method == "nmtf":
        # If NMTF is used, we run the NMTF function
        nmf_output = nmtf(in_mat, rank_factor=rank_factor, norm_thresh=norm_thresh, zero_threshold=zero_threshold,
                    init_func=nmtf_initialization_random,topic_count=topic_count)
    else:
        raise ValueError(f"Unknown NMF method: {nmf_method}. Supported methods are: 'nmf', 'pnmf', 'nmtf'")

    if nmf_output is None:
        raise RuntimeError(f"NMF computation failed for method: {nmf_method}")

    in_mat = in_mat.tocsr()

    w = sp.csr_matrix(w)
    h = sp.csc_matrix(h)

    return nmf_output


def run_nmf(num_of_topics: int, sparse_matrix: scipy.sparse.csr.csr_matrix, init = nmf_initialization_nndsvd, norm_thresh=0.001, zero_threshold=0.00001,nmf_method:str = "nmf"):
    """
    Performs Non-negative Matrix Factorization (NMF) on the given sparse matrix to decompose it into two non-negative
    matrices W and H. This method is appropriate for dimensionality reduction and identifying latent factors
    in the data. The process takes into account initialization methods, normalization thresholds, zero thresholds,
    and the specific NMF algorithm to use.

    :param num_of_topics: The number of topics or latent factors to decompose the matrix into.
    :type num_of_topics: int
    :param sparse_matrix: The input data matrix in compressed sparse row format that is to be factorized.
    :type sparse_matrix: scipy.sparse.csr.csr_matrix
    :param init: A callable for initialization of the factor matrices (default is nmf_initialization_nndsvd).
    :type init: Callable
    :param norm_thresh: The threshold for normalization, adjusts stopping criteria for the algorithm (default is 0.001).
    :type norm_thresh: float, optional
    :param zero_threshold: A small threshold to determine approximately zero values in the matrices (default is 0.00001).
    :type zero_threshold: float, optional
    :param nmf_method: A flag or method type signifying which NMF computation to run ["opnmf","nmf"] (default is "nmf").
    :type nmf_method: str
    :return: A tuple containing two matrices W and H. W represents the basis matrix, and H represents the coefficient matrix.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    sparse_matrix = sparse_matrix.tocsc()
    
    # convert all matrices to dense with toarray()
    
    outputs = _nmf_cpu(sparse_matrix,
                    log=True,
                    rank_factor=1.0,
                    norm_thresh=norm_thresh,
                    zero_threshold=zero_threshold,
                    init_func=init,
                    topic_count=num_of_topics,
                    nmf_method = nmf_method
                    )
    
    # convert all matrices to dense with toarray()

    return outputs