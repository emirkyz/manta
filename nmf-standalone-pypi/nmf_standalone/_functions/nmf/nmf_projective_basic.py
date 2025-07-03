from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.sparse import coo_matrix

def projective_nmf(X: sp.csr_matrix, r: int, options: Optional[Dict] = None, init: bool = None, W_mat: sp.csr_matrix = None) -> Tuple[np.ndarray, np.ndarray]:
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
    display = False
    if display:
        print('Running projective NMF (sparse):')

    # Initialize variables
    i = 0
    e = np.zeros(maxiter)
    W = W_mat
    while i < maxiter:
        old_w = W
        XtW = X.T @ W
        XXtW = X @ XtW  # m by r (dense)
        WtW = W.T @ W  # r by r (dense)
        XtWtXtW = XtW.T @ XtW  # r by r (dense)
        # temp_pay = X @ XtW  # m by r (dense)
        #pay = XXt @ W  # m by r (dense)

        #payda = W @ (W.T @ (pay))

        W = W * (X @ (X.T @ W)) / (W @ (W.T @ (X @ (X.T @ W))) + 1e-10)  # Avoid division by zero
        W = W / np.linalg.norm(W, ord=2)  # Normalize W to prevent numerical issues

        # alpha = np.sum(X) / np.sum(W @ (W.T @ (X)))  # Scaling factor
        # W = W * np.sqrt(alpha)

        '''# Sparse matrix multiplications
        XtW = X.T @ W      # n by r (dense)
        XXtW = X @ XtW     # m by r (dense)

        # Optimal scaling of the initial solution
        W = W / np.linalg.norm(W, ord=2)

        # MU by Yang and Oja
        W = W * (XXtW) / (W @ (W.T @ (XXtW)) + 1e-10)'''
        w_delta = np.linalg.norm(np.abs(W - old_w), 'fro')

        print(f"Iteration {i+1}, norm change: {w_delta:.4f}", end='\r')
        if w_delta < delta:
            if display:
                print(f"\nConverged after {i+1} iterations.")
            break

        i += 1
    if display:
        print()
        
    h = W.T @ X

    return W, h
