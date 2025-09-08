import numpy as np
from datetime import datetime, timedelta
from typing import Callable

import scipy as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm
def _basic_nmf(in_mat:sp.sparse.csc_matrix, w, h, start, log: bool = True, norm_thresh=0.005, zero_threshold=0.0001,
             norm_func: Callable = np.linalg.norm) -> tuple:
    """
    This function is the core of the NMF algorithm.
    Takes a sparse matrix, a W matrix, a H matrix, a start time, a log flag, a norm threshold, a zero threshold and a norm function and returns the W and H matrices.
    
    Args:
        in_mat: sparse matrix
        w: W matrix
        h: H matrix
        start: start time
        log: log flag
        norm_thresh: norm threshold; default is 0.005; if the norm of the W or H matrix is less than this, the algorithm stopstops
        norm_func: norm function; default is np.linalg.norm; can be np.linalg.norm or np.linalg.norm2
    Returns:
        w: W matrix. Shape is (m, r) where m is the number of rows in the input matrix and r is the number of topics.
        h: H matrix. Shape is (r, n) where n is the number of columns in the input matrix.
    """
    i = 0
    # check if w or h is zero
    max_iter = 10_000
    eps = 1e-10
    #obj = np.inf
    log = True
    pbar = tqdm(total=max_iter, desc="NMF iterations") if log else None

    #np.show_config()
    while True:
        w_old = w
        h_old = h



        wT_w = w.T @ w
        numerator_h = w.T @ in_mat
        denominator_h = wT_w @ h + eps
        h = h * (numerator_h / denominator_h)

        h_hT = h @ h.T
        numerator_w = in_mat @ h.T
        denominator_w = w @ h_hT + eps
        w = w * (numerator_w / denominator_w)

        #w1 = w * ((in_mat @ h.T) / (w @ (h @ h.T) + eps))
        #h1 = h * ((w1.T @ in_mat) / ((w1.T @ w1) @ h + eps))

        w_norm = norm_func(np.abs(w - w_old), 2)
        h_norm = norm_func(np.abs(h - h_old), 2)
        if log and pbar:
            pbar.update(1)
            pbar.set_postfix({"W_norm": f"{w_norm:.5f}", "H_norm": f"{h_norm:.5f}"})
        if i >= max_iter:
            if log:
                print('\n', 'Max iteration reached, giving up...')
            break

        if w_norm < norm_thresh and h_norm < norm_thresh:
            print('\nConvergence threshold achieved.')
            break

        ''' 
        e = 0.5
        w1[w1 < e] = e
        h1[h1 < e] = e
       
        # square of euclidean distance
        divergence_x_xnew = norm_func(np.abs(in_mat-(w1@h1)), 2)
        d_delta = divergence_x_xnew - obj

        if d_delta < norm_thresh:
            if log:
                print('\n', 'Requested Norm Threshold achieved, giving up...')
            break

        obj = norm_func(np.abs(in_mat-(w1@h1)), 2)
        '''

        i += 1

        
    w[w < zero_threshold] = 0
    h[h < zero_threshold] = 0

    nmf_output = {}
    nmf_output["W"] = w
    nmf_output["H"] = h
    
    return nmf_output