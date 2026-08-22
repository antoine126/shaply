"""Dependency-free clustering helpers (numpy only) shared by advanced plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.intp]


def average_linkage_order(distance: FloatArray) -> list[int]:
    """Return a leaf ordering from average-linkage agglomerative clustering.

    Produces a seriation of the items so that similar rows/columns sit next to
    each other, suitable for reordering a similarity heatmap. Runs in ``O(n^3)``,
    which is ample for the handful of features a plot displays.

    Parameters
    ----------
    distance
        Square, symmetric pairwise distance matrix.

    Returns
    -------
    list of int
        Item indices in clustered order.
    """
    n = distance.shape[0]
    if n <= 1:
        return list(range(n))

    size = 2 * n
    dist = np.full((size, size), np.inf)
    dist[:n, :n] = distance
    np.fill_diagonal(dist, np.inf)

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    counts: dict[int, int] = dict.fromkeys(range(n), 1)
    active = set(range(n))
    next_id = n

    while len(active) > 1:
        active_ids = list(active)
        sub = dist[np.ix_(active_ids, active_ids)]
        flat = int(np.argmin(sub))
        ia, ja = divmod(flat, len(active_ids))
        i, j = active_ids[ia], active_ids[ja]

        members[next_id] = members[i] + members[j]
        counts[next_id] = counts[i] + counts[j]
        for m in active:
            if m in (i, j):
                continue
            merged = (counts[i] * dist[i, m] + counts[j] * dist[j, m]) / (counts[i] + counts[j])
            dist[next_id, m] = dist[m, next_id] = merged

        active.discard(i)
        active.discard(j)
        active.add(next_id)
        next_id += 1

    return members[next(iter(active))]


def kmeans(
    x: FloatArray,
    n_clusters: int,
    *,
    random_state: int = 0,
    n_iter: int = 100,
) -> tuple[IntArray, FloatArray]:
    """Cluster rows of ``x`` with a compact k-means++ / Lloyd implementation.

    Parameters
    ----------
    x
        Data matrix, shape ``(n_samples, n_features)``.
    n_clusters
        Number of clusters (clamped to at most ``n_samples``).
    random_state
        Seed for reproducible initialization.
    n_iter
        Maximum number of Lloyd iterations.

    Returns
    -------
    labels : numpy.ndarray
        Cluster index per sample, shape ``(n_samples,)``.
    centers : numpy.ndarray
        Cluster centroids, shape ``(n_clusters, n_features)``.
    """
    rng = np.random.default_rng(random_state)
    n_samples = x.shape[0]
    k = min(n_clusters, n_samples)

    # k-means++ initialization.
    centers = [x[rng.integers(n_samples)]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        total = dist_sq.sum()
        probs = dist_sq / total if total > 0 else np.full(n_samples, 1.0 / n_samples)
        centers.append(x[rng.choice(n_samples, p=probs)])
    centroids = np.array(centers, dtype=np.float64)

    labels = np.zeros(n_samples, dtype=np.intp)
    for _ in range(n_iter):
        distances = np.stack([np.sum((x - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(distances, axis=1).astype(np.intp)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = x[labels == c]
            if members.size:
                centroids[c] = members.mean(axis=0)
    return labels, centroids
