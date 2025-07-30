import math
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import scipy.sparse as sp

from manta._functions.nmf.nmtf.extract_nmtf_topics import extract_topics

from .nmtf_init import nmtf_initialization_random
from .nmtf_util import sort_matrices

def _calculate_rank_range(m: int, n: int) -> tuple[int, int]:
    delta = (m + n) ** 2 + 4 * m * n
    x1, x2 = (-(m + n) + math.sqrt(delta)) * 0.5, (-(m + n) - math.sqrt(delta)) * 0.5
    x1, x2 = int(math.ceil(x1)), int(math.ceil(x2))
    x1, x2 = min(x1, x2), max(x1, x2)
    return max(x1, 0), max(x2, 2)


def _calculate_rank_range_sparse(m: int, n: int, nnz: int) -> tuple[int, int]:
    delta = (m + n) ** 2 + 4 * nnz
    x1, x2 = (-(m + n) + math.sqrt(delta)) * 0.5, (-(m + n) - math.sqrt(delta)) * 0.5
    x1, x2 = int(math.ceil(x1)), int(math.ceil(x2))
    x1, x2 = min(x1, x2), max(x1, x2)
    return max(x1, 0), max(x2, 2)


def _nmtf(in_mat: sp.csc_matrix, log: bool = True, rank_factor: float = 1.0,
          norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
          init_func: Callable = nmtf_initialization_random) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csc_matrix]:
    m, n = in_mat.shape
    # k_range = _calculate_rank_range_sparse(m, n, in_mat.nnz)
    # therotical_max_value = k_range[1]
    # target_rank = int(therotical_max_value * rank_factor)
    target_rank = 10
    w, s, h = init_func(in_mat, target_rank)

    if log:
        print("Performing NMTF...")
        start = datetime.now()

    w, s, h = _core_nmtf(in_mat, w, s, h, start, log=log, norm_thresh=norm_thresh, zero_threshold=zero_threshold,
                         norm_func=np.linalg.norm)

    w = sp.csr_matrix(w)
    s = sp.csr_matrix(s)
    h = sp.csc_matrix(h)

    #ind, max_vals = sort_matrices(s.toarray())

    #extract_topics(w, h, doc_word_pairs=ind, weights=max_vals)
    
    return w, s, h


def _core_nmtf(in_mat, w, s, h, start, log: bool = True, norm_thresh=1.0, zero_threshold=0.000001,
               norm_func: Callable = np.linalg.norm) -> tuple:
    i = 0
    while True:
        w1 = w * ((in_mat @ (h.T @ s.T)) / (w @ s @ (h @ h.T) @ s.T + 1e-10))
        s1 = s * ((w1.T @ in_mat @ h.T) / ((w1.T @ w1) @ s @ (h @ h.T) + 1e-10))
        h1 = h * ((s1.T @ (w1.T @ in_mat)) / (s1.T @ (w1.T @ w1) @ s1 @ h + 1e-10))

        w_norm = norm_func(np.abs(w1 - w), 2)
        h_norm = norm_func(np.abs(h1 - h), 2)
        s_norm = norm_func(np.abs(s1 - s), 2)
        if log:
            duration = datetime.now() - start
            duration_sec = round(duration.total_seconds())
            duration = timedelta(seconds=duration_sec)
            if duration_sec == 0:
                print(f"{i + 1}. step L2 W: {w_norm:.5f} S: {s_norm:.5f} H: {h_norm:.5f}. Duration: {duration}.",
                      end='\r')
            else:
                print(f"{i + 1}. step L2 W: {w_norm:.5f} S: {s_norm:.5f} H: {h_norm:.5f}. Duration: {duration}. "
                      f"Speed: {round((i + 1) * 24 / duration_sec, 2):.2f} matrix multiplications/sec", end='\r')
        w = w1
        h = h1
        s = s1
        i += 1

        if w_norm < norm_thresh and h_norm < norm_thresh and s_norm < norm_thresh:
            if log:
                print('\n', 'Requested Norm Threshold achieved, giving up...')
            break
        if i > 1000:
            if log:
                print('\n', 'Maximum iteration count reached, giving up...')
            break
    w[w < zero_threshold] = 0
    h[h < zero_threshold] = 0
    s[s < zero_threshold] = 0
    

    return w, s, h


def nmtf(in_mat: sp.csc_matrix, log: bool = True, rank_factor: float = 1.0,
         norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
         init_func: Callable = nmtf_initialization_random) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csc_matrix]:
    
    w, s, h = _nmtf(in_mat, log, rank_factor, norm_thresh, zero_threshold, init_func)
    
    nmf_output = {}
    nmf_output["W"] = w
    nmf_output["S"] = s

    nmf_output["H"] = h
    
    return nmf_output
