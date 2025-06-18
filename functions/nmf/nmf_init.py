import math

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sla


def select_rank_theoretical(in_mat: sp.csc_matrix) -> int:
    m, n = in_mat.shape
    return int(math.ceil(in_mat.nnz / (m + n)))


def select_rank_by_svd(in_mat: sp.csc_matrix) -> int:
    m, n = in_mat.shape
    target_svd_k = min(m, n)
    u, s, v = sla.svds(in_mat, k=target_svd_k)
    # TODO Burada kaldın.


def nmf_initialization_nndsvd(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray]:
    if rank < 0:
        rank = select_rank_theoretical(in_mat)
    print(f"Using Rank : {rank}")
    u, s, v = sla.svds(in_mat, k=rank)
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    v = v[idx, :]

    w = np.zeros((in_mat.shape[0], rank))
    h = np.zeros((rank, in_mat.shape[1]))
    w[:, 0] = math.sqrt(s[0]) * abs(u[:, 0])
    h[0, :] = math.sqrt(s[0]) * abs(v[0, :].T)
    for i in range(1, rank):
        uu = u[:, i]
        vv = v[i, :]
        uup = np.multiply(uu >= 0, uu)
        uun = np.multiply(uu < 0, -uu)
        vvp = np.multiply(vv >= 0, vv)
        vvn = np.multiply(vv < 0, -vv)
        n_uup = np.linalg.norm(uup, 2)
        n_uun = np.linalg.norm(uun, 2)
        n_vvp = np.linalg.norm(vvp, 2)
        n_vvn = np.linalg.norm(vvn, 2)
        termp = n_uup * n_vvp
        termn = n_uun * n_vvn
        if termp >= termn:
            w[:, i] = math.sqrt(s[i] * termp) / n_uup * uup
            h[i, :] = math.sqrt(s[i] * termp) / n_vvp * vvp.T
        else:
            w[:, i] = math.sqrt(s[i] * termn) / n_uun * uun
            h[i, :] = math.sqrt(s[i] * termn) / n_vvn * vvn.T
    w[w < 1e-11] = 0
    h[h < 1e-11] = 0
    return w, h


def nmf_initialization_random(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray]:
    if rank < 0:
        rank = select_rank_theoretical(in_mat)
    min_v = in_mat.min()
    max_v = in_mat.max()

    m, n = in_mat.shape

    return np.random.uniform(min_v, max_v, (m, rank)), np.random.uniform(min_v, max_v, (rank, n))
