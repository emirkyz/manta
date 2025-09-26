from datetime import datetime, timedelta
from typing import Callable

import numpy as np
from scipy import sparse as sp

from ..nmf.nmf_initialization import nmf_initialization_random
from ..nmf.nmf_projective_basic import projective_nmf

def _nmf_cpu(in_mat: sp.csc_matrix, log: bool = True, rank_factor: float = 1.0,
             norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
             init_func: Callable = nmf_initialization_random,
             konu_sayisi=-1) -> tuple[sp.csr_matrix, sp.csc_matrix]:
    """
    Internal CPU-based NMF implementation using projective NMF.
    
    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        log (bool): Enable progress logging
        rank_factor (float): Rank factor for matrix factorization
        norm_thresh (float): Convergence threshold
        zero_threshold (float): Threshold for zeroing small values
        init_func (Callable): Matrix initialization function
        konu_sayisi (int): Number of topics/components
    
    Returns:
        tuple[sp.csr_matrix, sp.csc_matrix]: W and H matrices from NMF decomposition
    """
    # target_rank = math.floor(rank_func(in_mat) * rank_factor)

    w, h = init_func(in_mat, konu_sayisi)

    if log:
        print("Performing NMF...")
        start = datetime.now()


    #w, h = _core_nmf(in_mat, w, h, start, log=log, norm_thresh=norm_thresh, zero_threshold=zero_threshold,
    #                 norm_func=np.linalg.norm)
    w, h = projective_nmf(in_mat, 10, init=True, W_mat=w)
    w = sp.csr_matrix(w)
    h = sp.csc_matrix(h)

    return w, h


def _core_nmf(in_mat, w, h, start, log: bool = True, norm_thresh=0.005, zero_threshold=0.0001,
              norm_func: Callable = np.linalg.norm) -> tuple:
    """
    Core NMF iteration algorithm using multiplicative update rules.
    
    Args:
        in_mat: Input matrix to factorize
        w: W matrix (document-topic)
        h: H matrix (topic-word)
        start: Start time for logging
        log (bool): Enable progress logging
        norm_thresh (float): Convergence threshold
        zero_threshold (float): Threshold for zeroing small values
        norm_func (Callable): Norm function for convergence checking
    
    Returns:
        tuple: Updated (w, h) matrices
    """
    i = 0
    # check if w or h is zero
    eps = 1e-10
    while True:

        w1 = w * ((in_mat @ h.T) / (w @ (h @ h.T) + eps))
        h1 = h * ((w1.T @ in_mat) / ((w1.T @ w1) @ h + eps))

        w_norm = norm_func(np.abs(w1 - w), 2)
        h_norm = norm_func(np.abs(h1 - h), 2)
        if log:
            duration = datetime.now() - start
            duration_sec = round(duration.total_seconds())
            duration = timedelta(seconds=duration_sec)
            if duration_sec == 0:
                print(f"{i + 1}. step L2 W: {w_norm:.5f} H: {h_norm:.5f}. Duration: {duration}.", end='\r')
            else:
                print(f"{i + 1}. step L2 W: {w_norm:.5f} H: {h_norm:.5f}. Duration: {duration}. "
                      f"Speed: {round((i + 1) * 10 / duration_sec, 2):.2f} matrix multiplications/sec", end='\r')
        if i >= 1000:
            if log:
                print('\n', 'Max iteration reached, giving up...')
            break
        w = w1
        h = h1
        i += 1

        if w_norm < norm_thresh and h_norm < norm_thresh:
            if log:
                print('\n', 'Requested Norm Threshold achieved, giving up...')
            break
    w[w < zero_threshold] = 0
    h[h < zero_threshold] = 0
    return w, h


def nmf(in_mat: sp.csc_matrix, log: bool = True, rank_factor: float = 1.0,
        norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
        init_func: Callable = nmf_initialization_random):
    """
    Public wrapper function for NMF matrix factorization.
    
    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        log (bool): Enable progress logging
        rank_factor (float): Rank factor for matrix factorization
        norm_thresh (float): Convergence threshold
        zero_threshold (float): Threshold for zeroing small values
        init_func (Callable): Matrix initialization function
    
    Returns:
        tuple: W and H matrices from NMF decomposition
    """
    return _nmf_cpu(in_mat, log, rank_factor, norm_thresh, zero_threshold, init_func)
