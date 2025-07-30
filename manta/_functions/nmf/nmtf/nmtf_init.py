import numpy as np
import scipy.sparse as sp


def nmtf_initialization_random(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_v = in_mat.min()
    max_v = in_mat.max()

    m, n = in_mat.shape

    return np.random.uniform(min_v, max_v, (m, rank)), np.random.uniform(min_v, max_v, (rank, rank)), \
        np.random.uniform(min_v, max_v, (rank, n))


def nmtf_initialization_nndsvd(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from manta._functions.nmf.nmf_initialization import nmf_initialization_nndsvd

    wt, ht = nmf_initialization_nndsvd(in_mat, rank + 1)
    wt_sp = sp.csc_matrix(wt)
    ht_sp = sp.csc_matrix(ht)
    w, s_w = nmf_initialization_nndsvd(wt_sp, rank)
    s_h, h = nmf_initialization_nndsvd(ht_sp, rank)

    s = np.sqrt(s_w @ s_h)

    return w, s, h
