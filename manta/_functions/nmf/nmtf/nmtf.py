import math
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from ....utils.console.console_manager import ConsoleManager, get_console
from .nmtf_init import nmtf_initialization_random, nmtf_initialization_nndsvd, nmtf_initialization_nndsvd_direct, \
    nmtf_initialization_nndsvd_symmetric, nmtf_initialization_nndsvd_adaptive, nmtf_initialization_nndsvd_legacy


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


def _nmtf(in_mat: sp.csr_matrix, log: bool = True, rank_factor: float = 1.0,
          norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
          init_func: Callable = nmtf_initialization_random, konu_sayisi=10,
          method: str = "multiplicative", console: Optional[ConsoleManager] = None) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    _console = console or get_console()
    m, n = in_mat.shape
    # k_range = _calculate_rank_range_sparse(m, n, in_mat.nnz)
    # therotical_max_value = k_range[1]
    # target_rank = int(therotical_max_value * rank_factor)
    target_rank = konu_sayisi
    w, s, h = init_func(in_mat, target_rank)

    start = datetime.now()
    if log:
        _console.print_debug(f"Performing NMTF using {method} method...", tag="NMTF")

    if method == "multiplicative":
        w, s, h = _core_nmtf_test(in_mat, w, s, h, start, log=log, norm_thresh=norm_thresh, zero_threshold=zero_threshold,
                             norm_func=np.linalg.norm, console=_console)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'multiplicative', 'coordinate_descent', or 'projected_gradient'.")

    w = sp.csr_matrix(w)
    s = sp.csr_matrix(s)
    h = sp.csr_matrix(h)

    #ind, max_vals = sort_matrices(s.toarray())

    #extract_topics(w, h, doc_word_pairs=ind, weights=max_vals)
    
    return w, s, h


def s_matrix_confusion(s_matrix, console: Optional[ConsoleManager] = None):
    _console = console or get_console()
    ind = []
    max_values = []

    for i in range(s_matrix.shape[1]):
        col = s_matrix[:, i]
        max_ind = np.argmax(col)
        max_values.append(col[max_ind])
        ind.append((i, max_ind))

    ind_sorted = np.argsort(max_values)[::-1]
    ind = np.array(ind)[ind_sorted]
    max_values = np.array(max_values)[ind_sorted]

    for index, value in enumerate(max_values):
        _console.print_debug(f"Topic {index} has connection with Topic {ind[index][1]}, strength: {value}", tag="S-MATRIX")

    return



def _core_nmtf(in_mat, w, s, h, start, log: bool = True, norm_thresh=1.0, zero_threshold=0.0001,
               norm_func: Callable = np.linalg.norm, console: Optional[ConsoleManager] = None) -> tuple:
    _console = console or get_console()
    i = 0
    epsilon = 1e-9
    while True:
        w1 = w * ((in_mat @ (h.T @ s.T)) / ((w @ s @ (h @ h.T) @ s.T) + epsilon))
        s1 = s * ((w1.T @ in_mat @ h.T) / ((w1.T @ w1) @ s @ (h @ h.T)+ epsilon))
        h1 = h * ((s1.T @ (w1.T @ in_mat)) / (s1.T @ (w1.T @ w1) @ s1 @ h) + epsilon)

        w_norm = norm_func(np.abs(w1 - w), 2)
        h_norm = norm_func(np.abs(h1 - h), 2)
        s_norm = norm_func(np.abs(s1 - s), 2)
        if log:
            duration = datetime.now() - start
            duration_sec = round(duration.total_seconds())
            duration = timedelta(seconds=duration_sec)
            if duration_sec == 0:
                _console.print_debug(f"{i + 1}. step L2 W: {w_norm:.5f} S: {s_norm:.5f} H: {h_norm:.5f}. Duration: {duration}.", tag="NMTF")
            else:
                _console.print_debug(f"{i + 1}. step L2 W: {w_norm:.5f} S: {s_norm:.5f} H: {h_norm:.5f}. Duration: {duration}. "
                      f"Speed: {round((i + 1) * 24 / duration_sec, 2):.2f} matrix multiplications/sec", tag="NMTF")
        w = w1
        h = h1
        s = s1
        i += 1

        if w_norm < norm_thresh and h_norm < norm_thresh and s_norm < norm_thresh:
            if log:
                _console.print_debug("Requested Norm Threshold achieved, giving up...", tag="NMTF")
            break
    w[w < zero_threshold] = 0
    h[h < zero_threshold] = 0
    s[s < zero_threshold] = 0

    return w, s, h



def _core_nmtf_test(in_mat, w, s, h, start, log: bool = True, norm_thresh=1.0,
               zero_threshold=0.000001, norm_func: Callable = np.linalg.norm,
               console: Optional[ConsoleManager] = None) -> tuple:
    _console = console or get_console()
    i = 0
    epsilon = 1e-9  # Slightly larger epsilon for better numerical stability
    """
    # Normalize W columns
    w_col_norms = norm_func(w, axis=0, keepdims=True)
    w_col_norms[w_col_norms == 0] = epsilon  # Avoid division by zero
    w = w / w_col_norms
#
    # Normalize H rows
    h_row_norms = norm_func(h, axis=1, keepdims=True)
    h_row_norms[h_row_norms == 0] = epsilon  # Avoid division by zero
    h = h / h_row_norms
#

    # Normalize S diagonal
    s_diag_norms = norm_func(s, axis=1, keepdims=True)
    s_diag_norms[s_diag_norms == 0] = epsilon
    s = s / s_diag_norms
    # Transfer scales to S using matrix multiplication with diagonal matrices
    # S_new = diag(w_col_norms) @ S @ diag(h_row_norms)
    #s = np.diag(w_col_norms.flatten()) @ s @ np.diag(h_row_norms.flatten())
    """
    while True:
        # Update W
        numerator_w = in_mat @ (h.T @ s.T)
        denominator_w = w @ s @ (h @ h.T) @ s.T + epsilon
        w1 = w * (numerator_w / denominator_w)

        # Update S
        numerator_s = (w1.T @ in_mat @ h.T)
        denominator_s = (w1.T @ w1) @ s @ (h @ h.T) + epsilon
        s1 = s * (numerator_s / denominator_s)

        # Update H
        numerator_h = s1.T @ (w1.T @ in_mat)
        denominator_h = s1.T @ (w1.T @ w1) @ s1 @ h + epsilon
        h1 = h * (numerator_h / denominator_h)

        # # Normalize W columns
        # w_col_norms = norm_func(w1, axis=0, keepdims=True)
        # w_col_norms[w_col_norms == 0] = 1  # Avoid division by zero
        # w1 = w1 / w_col_norms
        # # Normalize H rows
        # h_row_norms = norm_func(h1, axis=1, keepdims=True)
        # h_row_norms[h_row_norms == 0] = 1  # Avoid division by zero
        # h1 = h1 / h_row_norms
        #
        # # Transfer scales to S using matrix multiplication with diagonal matrices
        # # S_new = diag(w_col_norms) @ S @ diag(h_row_norms)
        # s1 = np.diag(w_col_norms.flatten()) @ s1 @ np.diag(h_row_norms.flatten())

        ## Calculate convergence metrics
        w_norm = norm_func(np.abs(w1 - w), "fro")
        h_norm = norm_func(np.abs(h1 - h), "fro")
        s_norm = norm_func(np.abs(s1 - s), "fro")

        if log:
            duration = datetime.now() - start
            duration_sec = round(duration.total_seconds())
            duration = timedelta(seconds=duration_sec)
            if duration_sec == 0:
                _console.print_debug(f"{i + 1}. step L2 W: {w_norm:.5f} S: {s_norm:.5f} H: {h_norm:.5f}. Duration: {duration}.", tag="NMTF")
            else:
                _console.print_debug(f"{i + 1}. step L2 W: {w_norm:.5f} S: {s_norm:.5f} H: {h_norm:.5f}. Duration: {duration}. "
                      f"Speed: {round((i + 1) * 24 / duration_sec, 2):.2f} matrix multiplications/sec", tag="NMTF")

        # Update matrices
        w = w1
        h = h1
        s = s1
        i += 1

        # Check convergence
        if w_norm < norm_thresh and h_norm < norm_thresh and s_norm < norm_thresh:
            if log:
                _console.print_debug("Requested Norm Threshold achieved, stopping...", tag="NMTF")
            break
        if i > 1000:
            if log:
                _console.print_debug("Maximum iteration count reached, stopping...", tag="NMTF")
            break

    # Apply thresholding
    w[w < zero_threshold] = 0
    h[h < zero_threshold] = 0
    s[s < zero_threshold] = 0

    s_matrix_confusion(s, console=_console)
    _console.print_debug(f"S matrix shape: {s.shape}", tag="NMTF")
    return w, s, h


def nmtf(in_mat: sp.csr_matrix, log: bool = True, rank_factor: float = 1.0,
         norm_thresh: float = 1.0, zero_threshold: float = 0.0001,
         init_func: Callable = nmtf_initialization_random,
         topic_count:int = 10, method: str = "multiplicative") -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:

    w, s, h = _nmtf(in_mat, log, rank_factor, norm_thresh, zero_threshold, nmtf_initialization_nndsvd_legacy, topic_count, method)
    
    nmf_output = {}
    nmf_output["W"] = w.toarray()
    nmf_output["S"] = s.toarray()

    nmf_output["H"] = h.toarray()
    
    return nmf_output
