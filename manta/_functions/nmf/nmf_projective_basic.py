from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import normalize

def projective_nmf(X: sp.csc_matrix, r: int, options: Optional[Dict] = None, init: bool = None,
                   W_mat: sp.csr_matrix = None, h_mat: sp.csr_matrix = None,
                   norm_func: Callable = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform projective Non-negative Matrix Factorization (NMF) with multiplicative
    updates, especially tailored for sparse matrices. The function aims to factorize
    the input matrix into two non-negative matrices by iteratively updating the
    variables to minimize the reconstruction error.
    min_{W >= 0} ||X - WW^TX||_F^2
    :param X: Input sparse matrix of shape (m, n).
    :param r: Rank of the factorization (number of components).
    :param options: Dictionary containing optional configurations such as:
                    - 'maxiter' (int): Maximum number of iterations.
                    - 'delta' (float): Convergence threshold for changes in W.
                    - 'display' (bool): Whether to display iteration progress.
    :param init: (Optional) Indicates if the initialization is required.
    :param W_mat: Optional initial value for the W matrix with shape (m, r).
                  If provided, it will be used as the starting point for iteration.
    :return: A tuple containing:
             - W (np.ndarray): Basis matrix with shape (m, r).
             - h (np.ndarray): Coefficient matrix with shape (r, n).
    """
    print("Projective NMF starting...")
    m, n = X.shape

    # Set default options
    if options is None:
        options = {}

    maxiter = options.get('maxiter', 1000)
    delta = options.get('delta', 0.005)
    display = options.get('display', True)
    display = True
    if display:
        print('Running projective NMF (sparse):')
    zero_threshold = options.get('zero_threshold', 1e-10)
    # Initialize variables
    start = datetime.now()
    i = 0
    e = np.zeros(maxiter)

    W = W_mat
    eps = 1e-10
    log = True
    
    # Precompute X @ X.T once (expensive operation)
    '''
    \min_{W \geq 0} \|V - WW^T V\|
    
    
    Update rule for W:
      W_{ij} \leftarrow W_{ij} \frac{\sum_k \left((W^T V)_{jk} + \sum_l W_{lj} V_{ik}\right)}{\sum_k V_{ik} \left((W^T V)_{jk} / (WW^T V)_{ik} + \sum_l W_{lj} V_{lk} / (WW^T V)_{lk}\right)}
    '''
    while i < maxiter:
        # Multiplicative update rule for Projective NMF.
        # W_new = W * sqrt( (X @ X.T @ W) / (W @ W.T @ X @ X.T @ W) )
        # Yuan and Oja - 2005 - Projective Nonnegative Matrix Factorization for Image Compression and Feature Extraction.pdf eq:16

        # Optimized computation order to reduce memory strain
        # First compute smaller intermediate matrices (r × r) to reuse
        wtw = W.T @ W  # r × r matrix - much smaller, reusable
        vtw = X.T @ W  # n × r matrix
        
        # Compute the numerator (pay)
        pay = X @ vtw  # m × r matrix (this is vvtw)
        
        # Compute denominator components efficiently
        wt_pay = W.T @ pay  # r × r matrix - reuse smaller computation
        payda = W @ wt_pay + pay @ wtw + eps  # Both terms use precomputed r×r matrices
        
        # Element-wise division for the update
        w_new = W * np.sqrt(pay / payda)


        # Calculate the norm of the difference between the old and new W
        w_norm = norm_func(np.abs(w_new - W), 2)

        if log:
            duration = datetime.now() - start
            duration_sec = round(duration.total_seconds())
            duration = timedelta(seconds=duration_sec)
            if duration_sec == 0:
                print(f"{i + 1}. step L2 W: {w_norm:.5f}. Duration: {duration}.", end='\r')
            else:
                print(f"{i + 1}. step L2 W: {w_norm:.5f}. Duration: {duration}. "
                      f"Speed: {round((i + 1) / duration_sec, 2):.2f} iter/sec", end='\r')

        # --- Check for stopping conditions ---
        if i >= maxiter:
            if log:
                print('\n', 'Max iteration reached, giving up...')
            break

        if w_norm < delta:
            if log:
                print('\n', 'Requested Norm Threshold achieved, giving up...')
            break

        # Update W for the next iteration
        W = w_new
        i += 1

    # Apply zero threshold to clean up the final matrix
    W[W < zero_threshold] = 0

    # Final normalization
    w = normalize(W, norm='l2', axis=0)
    h = w.T @ X
    
    nmf_output = {}
    nmf_output["W"] = w
    nmf_output["H"] = h
    
    return nmf_output