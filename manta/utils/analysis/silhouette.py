"""
Simplified Silhouette Score for NMF topic models.

Computes centroid-based silhouette in W-space (doc-topic matrix).
Time complexity: O(n × k) — fully vectorized, scales to millions of documents.
"""

from typing import Any, Dict

import numpy as np

from .dominant_topic import get_dominant_topics


def calculate_simplified_silhouette(W: np.ndarray) -> Dict[str, Any]:
    """
    Compute simplified silhouette score using cluster centroids.

    For each document i assigned to cluster c:
        a(i) = ||w_i - centroid[c]||₂          (intra-cluster distance)
        b(i) = min_{j ≠ c} ||w_i - centroid[j]||₂   (nearest other cluster)
        s(i) = (b(i) - a(i)) / max(a(i), b(i))       ∈ [-1, 1]

    Fully vectorized: avoids Python loops over documents.
    Memory: O(n × k) — e.g. 900k docs × 10 topics ≈ 72 MB float64.

    Documents with all-zero topic scores (label -1) are excluded.
    Topics with fewer than 2 assigned documents get score 0.0.

    Args:
        W: Document-topic matrix, shape (n_docs, n_topics). Dense or sparse.

    Returns:
        Dict with:
            average (float | None): Overall mean silhouette ∈ [-1, 1],
                or None if computation is not possible.
            per_topic (dict[str, float]): Mean silhouette per topic.
            n_assigned (int): Documents with a valid topic assignment.
            n_total (int): Total documents.
    """
    if hasattr(W, "toarray"):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)

    n_docs, n_topics = W.shape
    labels = get_dominant_topics(W)  # (n_docs,), -1 = unassigned

    assigned_mask = labels >= 0
    n_assigned = int(assigned_mask.sum())

    empty_result: Dict[str, Any] = {
        "average": None,
        "per_topic": {},
        "n_assigned": n_assigned,
        "n_total": n_docs,
    }

    if n_assigned == 0 or n_topics < 2:
        return empty_result

    W_asgn = W[assigned_mask]                   # (n_assigned, k)
    L_asgn = labels[assigned_mask].astype(int)  # (n_assigned,)

    # Compute centroid for each topic
    centroids = np.zeros((n_topics, n_topics), dtype=np.float64)
    topic_counts = np.zeros(n_topics, dtype=np.int64)
    for j in range(n_topics):
        mask_j = L_asgn == j
        topic_counts[j] = int(mask_j.sum())
        if topic_counts[j] > 0:
            centroids[j] = W_asgn[mask_j].mean(axis=0)

    valid_topics = np.where(topic_counts >= 1)[0]
    if len(valid_topics) < 2:
        return empty_result

    # Vectorized distance: each doc to each centroid
    # ||w_i - c_j||² = ||w_i||² + ||c_j||² - 2 w_i · c_j
    W_sq = (W_asgn ** 2).sum(axis=1)               # (n_assigned,)
    C_sq = (centroids ** 2).sum(axis=1)             # (n_topics,)
    cross = W_asgn @ centroids.T                    # (n_assigned, n_topics)
    del W_asgn  # no longer needed; free early
    dists_sq = W_sq[:, None] + C_sq[None, :] - 2.0 * cross
    del W_sq, C_sq, cross
    dists = np.sqrt(np.maximum(dists_sq, 0.0))     # (n_assigned, n_topics)
    del dists_sq

    # Mark invalid topics (0 docs) so they are never picked as b
    invalid_mask = topic_counts == 0               # (n_topics,)
    dists[:, invalid_mask] = np.inf

    row_idx = np.arange(n_assigned)

    # a(i) = save own-cluster distances, then mask in-place (avoids a full copy)
    a = dists[row_idx, L_asgn].copy()             # (n_assigned,)
    dists[row_idx, L_asgn] = np.inf

    # b(i) = min distance to any other centroid (dists already masked in-place)
    b = dists.min(axis=1)                          # (n_assigned,)
    del dists

    # s(i) = (b - a) / max(a, b); 0 when topic has < 2 docs
    denom = np.maximum(a, b)
    s = np.where(
        (topic_counts[L_asgn] >= 2) & (denom > 1e-12),
        (b - a) / denom,
        0.0,
    )

    # Per-topic mean silhouette
    per_topic: Dict[str, float] = {}
    for j in range(n_topics):
        mask_j = L_asgn == j
        if mask_j.sum() > 0:
            per_topic[f"topic_{j + 1:02d}"] = float(round(float(s[mask_j].mean()), 4))

    return {
        "average": float(round(float(s.mean()), 4)),
        "per_topic": per_topic,
        "n_assigned": n_assigned,
        "n_total": n_docs,
    }
