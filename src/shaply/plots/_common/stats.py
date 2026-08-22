"""Small dependency-free statistics helpers (numpy only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]


def average_ranks(x: FloatArray) -> FloatArray:
    """Return average (tie-corrected) ranks of ``x``."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def pearson(a: FloatArray, b: FloatArray) -> float:
    """Pearson correlation of two 1D arrays; ``0.0`` if a side has no variance."""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denom)


def spearman(a: FloatArray, b: FloatArray) -> float:
    """Spearman rank correlation of two 1D arrays (monotonic association)."""
    return pearson(average_ranks(a), average_ranks(b))
