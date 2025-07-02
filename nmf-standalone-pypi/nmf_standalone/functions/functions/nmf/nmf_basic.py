import numpy as np
from datetime import datetime, timedelta
from typing import Callable
from sklearn.preprocessing import normalize
def _basic_nmf(in_mat, w, h, start, log: bool = True, norm_thresh=0.005, zero_threshold=0.0001,
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
        if i >= max_iter:
            if log:
                print('\n', 'Max iteration reached, giving up...')
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