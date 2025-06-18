from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.sparse import coo_matrix


def projective_nmf(X: sp.csr_matrix, r: int, options: Optional[Dict] = None, init: bool = None,
                   W_mat: sp.csr_matrix = None, H_mat: sp.csr_matrix = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve projective NMF using multiplicative updates for sparse matrices:
        min_{W >= 0} ||X - WW^TX||_F^2

    Args:
        X: Sparse data matrix (csr_matrix)
        r: Factorization rank
        options: Dictionary containing optional parameters:
            - init: Initial matrix W
            - maxiter: Number of iterations (default: 200)
            - delta: Convergence threshold (default: 1e-4)
            - display: If True, displays iteration progress (default: True)

    Returns:
        W: Nonnegative matrix s.t. WW^TX is close to X
        e: Evolution of the relative error, ||X - WW^TX||_F/||X||_F
    """
    print("Projective NMF starting...")
    m, n = X.shape

    # Set default options
    if options is None:
        options = {}

    maxiter = options.get('maxiter', 5000)
    delta = options.get('delta', 0.005)
    display = options.get('display', True)
    display = True
    if display:
        print('Running projective NMF (sparse):')

    # Initialize variables
    # nX2 = X.power(2).sum()  # Efficient sparse sum of squares
    i = 0
    # e = np.zeros(maxiter)
    # Wp = np.zeros_like(W_mat)
    W = W_mat

    XXt = X @ X.T  # m by m (dense)
    start = datetime.now()
    while i < maxiter:

        old_w = W
        XtW = X.T @ W
        #temp_pay = X @ XtW  # m by r (dense)
        pay = XXt @ W # m by r (dense)


        payda = W @ (W.T @ (pay))

        W = W * (XXt @ W) / (W @ (W.T @ (XXt @ W)) + 1e-10) # Avoid division by zero
        # Ensure non-negativity

        # stabilization
        W = W / np.linalg.norm(W, ord=2)

        delta_w = np.linalg.norm(np.abs(W - old_w), ord="fro")

        duration = datetime.now() - start
        duration_sec = round(duration.total_seconds())
        print(f"Iteration {i + 1}, Delta W: {delta_w:.4f}", end='\r')
        if duration_sec != 0:
            print(f"Iteration {i + 1}, Delta W: {delta_w:.4f} "
                f"Speed: {round((i + 1) * 10 / duration_sec, 2):.2f} matrix multiplications/sec", end = '\r')
        if delta_w < delta:
            obj = np.linalg.norm(X - (W @ (W.T @ X)),
                                 ord="fro")  # calculating obj every iteration is too expensive, but we do it here for convergence check
            print(f"\nConvergence reached at iteration {i + 1}. Objective: {obj:.4f}")
            break

        i += 1

    H = W.T @ X  # r by n (dense)

    # mse = np.linalg.norm(X- (W @ H), ord='fro')**2 / (m * n)  # Mean Squared Error

    return W, H


'''
        Wp = W.copy()

        # Sparse matrix multiplications
        XtW = X.T @ W      # n by r (dense)
        XXtW = X @ XtW     # m by r (dense)
        WtW = W.T @ W      # r by r (dense)
        XtWtXtW = XtW.T @ XtW  # r by r (dense)

        # Optimal scaling of the initial solution
        alpha = np.sum(XXtW * W) / np.sum(WtW * XtWtXtW)
        W = W * np.sqrt(alpha)

        # Update the other factors accordingly
        XtW = np.sqrt(alpha) * XtW
        XXtW = np.sqrt(alpha) * XXtW
        WtW = alpha * WtW
        XtWtXtW = alpha * XtWtXtW

        # Calculate error (using sparse operations where possible)
        e[i] = np.sqrt(max(0, nX2 - 2*np.sum(XXtW * W) + np.sum(WtW * XtWtXtW)))

        # MU by Yang and Oja
        W = W * (2 * XXtW) / (W @ XtWtXtW + XXtW @ WtW + 1e-10)

        if display:
            print(f'{i+1}...', end='')
            if (i + 1) % 10 == 0:
                print()

        i += 1

    # Trim the error array to actual iterations performed
    e = e[:i] / sp.linalg.norm(X)

    if display:
        print()

    h = W.T @ X

    return W, h
    '''


def _opnmf_cpu(in_mat, w, h, konu_sayisi, start, log: bool = True, norm_thresh=0.005, zero_threshold=0.0001,
               norm_func: Callable = np.linalg.norm) -> tuple[sp.csr_matrix, sp.csc_matrix]:
    w, h = nmf_initialization_nndsvd(in_mat, konu_sayisi)
    while True:

        xxt = in_mat @ in_mat.T
        N = xxt @ w

        denominator_d = w @ w.T @ N

        pay = xxt @ w
        payda = denominator_d

        w_i = w * (pay / payda)

        w_norm = norm_func(np.abs(w_i - w), 2)

        print(w_norm)
        if w_norm < norm_thresh:
            break

        w = w_i

    h = w @ in_mat.T

    return w, h


