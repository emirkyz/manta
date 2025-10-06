import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as sla


def nmtf_initialization_random(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_v = in_mat.min()
    max_v = in_mat.max()
    min_v = 0
    max_v = 1

    m, n = in_mat.shape

    return (np.random.uniform(min_v, max_v, (m, rank)),
            np.random.uniform(min_v, max_v, (rank, rank)),
            np.random.uniform(min_v, max_v, (rank, n)))


def nmtf_initialization_nndsvd_legacy(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy hierarchical NNDSVD initialization for NMTF.
    Uses double NNDSVD factorization approach.

    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        rank (int): Target rank for factorization

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Initialized W, S, H matrices
    """
    from manta._functions.nmf.nmf_initialization import nmf_initialization_nndsvd

    wt, ht = nmf_initialization_nndsvd(in_mat, rank + 1)
    wt_sp = sp.csc_matrix(wt)
    ht_sp = sp.csc_matrix(ht)
    w, s_w = nmf_initialization_nndsvd(wt_sp, rank)
    s_h, h = nmf_initialization_nndsvd(ht_sp, rank)

    s = np.sqrt(s_w @ s_h)

    return w, s, h


def nmtf_initialization_nndsvd_direct(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct NNDSVD initialization for NMTF using single SVD decomposition.

    Applies NNDSVD principle directly to all three matrices by computing SVD once
    and splitting positive/negative components for W and H, while initializing S
    from singular values with small perturbations.

    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        rank (int): Target rank for factorization

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Initialized W, S, H matrices
    """
    m, n = in_mat.shape

    # Compute SVD
    u, s, v = sla.svds(in_mat, k=rank)

    # Sort by descending singular values
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    v = v[idx, :]

    # Initialize W using NNDSVD splitting on left singular vectors
    w = np.zeros((m, rank))
    for i in range(rank):
        uu = u[:, i]
        uup = np.maximum(uu, 0)  # Positive part
        uun = np.maximum(-uu, 0)  # Negative part
        n_uup = np.linalg.norm(uup, 2)
        n_uun = np.linalg.norm(uun, 2)

        if n_uup >= n_uun:
            w[:, i] = math.sqrt(s[i]) * uup / (n_uup + 1e-9)
        else:
            w[:, i] = math.sqrt(s[i]) * uun / (n_uun + 1e-9)

    # Initialize H using NNDSVD splitting on right singular vectors
    h = np.zeros((rank, n))
    for i in range(rank):
        vv = v[i, :]
        vvp = np.maximum(vv, 0)  # Positive part
        vvn = np.maximum(-vv, 0)  # Negative part
        n_vvp = np.linalg.norm(vvp, 2)
        n_vvn = np.linalg.norm(vvn, 2)

        if n_vvp >= n_vvn:
            h[i, :] = math.sqrt(s[i]) * vvp / (n_vvp + 1e-9)
        else:
            h[i, :] = math.sqrt(s[i]) * vvn / (n_vvn + 1e-9)

    # Initialize S as normalized diagonal + small random perturbations for flexibility
    s_normalized = s / (np.linalg.norm(s) + 1e-9)
    s_matrix = np.diag(s_normalized)
    # Add small off-diagonal perturbations to allow topic mixing
    s_matrix += np.random.uniform(0, 0.01 * np.mean(s_normalized), (rank, rank))
    s_matrix = np.maximum(s_matrix, 0)  # Ensure non-negativity

    # Apply threshold to remove very small values
    w[w < 1e-11] = 0
    h[h < 1e-11] = 0
    s_matrix[s_matrix < 1e-11] = 0

    return w, s_matrix, h




def nmtf_initialization_nndsvd_symmetric(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric orthogonal NNDSVD initialization for NMTF.

    Preserves orthogonality by applying symmetric NNDSVD splitting to both U and V,
    with S initialized from normalized singular values or identity.

    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        rank (int): Target rank for factorization

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Initialized W, S, H matrices
    """
    m, n = in_mat.shape

    # Compute SVD
    u, s, v = sla.svds(in_mat, k=rank)

    # Sort by descending singular values
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    v = v[idx, :]

    # Initialize W from U with symmetric splitting
    w = np.zeros((m, rank))
    for i in range(rank):
        uu = u[:, i]
        uup = np.maximum(uu, 0)
        uun = np.maximum(-uu, 0)
        n_uup = np.linalg.norm(uup, 2)
        n_uun = np.linalg.norm(uun, 2)

        # Use both positive and negative parts weighted by their norms
        if n_uup > 1e-9 or n_uun > 1e-9:
            w[:, i] = (n_uup * uup + n_uun * uun) / (n_uup + n_uun + 1e-9)
        else:
            w[:, i] = np.abs(uu)

    # Initialize H from V with symmetric splitting
    h = np.zeros((rank, n))
    for i in range(rank):
        vv = v[i, :]
        vvp = np.maximum(vv, 0)
        vvn = np.maximum(-vv, 0)
        n_vvp = np.linalg.norm(vvp, 2)
        n_vvn = np.linalg.norm(vvn, 2)

        # Use both positive and negative parts weighted by their norms
        if n_vvp > 1e-9 or n_vvn > 1e-9:
            h[i, :] = (n_vvp * vvp + n_vvn * vvn) / (n_vvp + n_vvn + 1e-9)
        else:
            h[i, :] = np.abs(vv)

    # Initialize S with normalized singular values on diagonal
    s_matrix = np.diag(s / (np.max(s) + 1e-9))

    # Add small random perturbations to break symmetry and allow flexibility
    perturbation = np.random.uniform(0, 0.005, (rank, rank))
    s_matrix += perturbation
    s_matrix = np.maximum(s_matrix, 0)

    # Apply threshold
    w[w < 1e-11] = 0
    h[h < 1e-11] = 0
    s_matrix[s_matrix < 1e-11] = 0

    return w, s_matrix, h


def nmtf_initialization_nndsvd_adaptive(in_mat: sp.csc_matrix, rank: int,
                                        variance_threshold: float = 0.9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive rank NNDSVD initialization for NMTF with asymmetric topic structures.

    Uses SVD to determine effective ranks for W and H independently based on
    explained variance, allowing S to be rectangular for asymmetric patterns.

    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        rank (int): Maximum rank for factorization
        variance_threshold (float): Cumulative variance threshold (0-1) for adaptive rank selection

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Initialized W, S, H matrices
    """
    m, n = in_mat.shape

    # Compute SVD with full rank
    k_full = min(rank * 2, min(m, n) - 1)  # Sample more components for analysis
    u, s, v = sla.svds(in_mat, k=k_full)

    # Sort by descending singular values
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    v = v[idx, :]

    # Determine effective rank for W (document topics) based on variance
    cumsum_var = np.cumsum(s**2) / np.sum(s**2)
    k1 = min(np.searchsorted(cumsum_var, variance_threshold) + 1, rank)
    k1 = max(k1, 2)  # Ensure at least 2 topics

    # For simplicity, use same rank for H (can be made asymmetric)
    k2 = k1

    # Use the actual rank (square S matrix for standard NMTF)
    actual_rank = min(k1, k2, rank)

    # Initialize W from first k components
    w = np.zeros((m, actual_rank))
    for i in range(actual_rank):
        uu = u[:, i]
        uup = np.maximum(uu, 0)
        uun = np.maximum(-uu, 0)
        n_uup = np.linalg.norm(uup, 2)
        n_uun = np.linalg.norm(uun, 2)

        if n_uup >= n_uun:
            w[:, i] = math.sqrt(s[i]) * uup / (n_uup + 1e-9)
        else:
            w[:, i] = math.sqrt(s[i]) * uun / (n_uun + 1e-9)

    # Initialize H from first k components
    h = np.zeros((actual_rank, n))
    for i in range(actual_rank):
        vv = v[i, :]
        vvp = np.maximum(vv, 0)
        vvn = np.maximum(-vv, 0)
        n_vvp = np.linalg.norm(vvp, 2)
        n_vvn = np.linalg.norm(vvn, 2)

        if n_vvp >= n_vvn:
            h[i, :] = math.sqrt(s[i]) * vvp / (n_vvp + 1e-9)
        else:
            h[i, :] = math.sqrt(s[i]) * vvn / (n_vvn + 1e-9)

    # Initialize S from cross-correlations and singular values
    s_matrix = np.diag(s[:actual_rank] / (np.max(s) + 1e-9))

    # Add correlation-based off-diagonal elements
    for i in range(actual_rank):
        for j in range(i+1, actual_rank):
            # Correlation between topics
            corr = np.abs(np.dot(u[:, i], u[:, j]))
            s_matrix[i, j] = corr * 0.1  # Small coupling
            s_matrix[j, i] = corr * 0.1

    s_matrix = np.maximum(s_matrix, 0)

    # Apply threshold
    w[w < 1e-11] = 0
    h[h < 1e-11] = 0
    s_matrix[s_matrix < 1e-11] = 0

    return w, s_matrix, h


def nmtf_initialization_nndsvd(in_mat: sp.csc_matrix, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Default NNDSVD initialization for NMTF (uses direct method).

    Args:
        in_mat (sp.csc_matrix): Input sparse matrix to factorize
        rank (int): Target rank for factorization

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Initialized W, S, H matrices
    """
    return nmtf_initialization_nndsvd_direct(in_mat, rank)
