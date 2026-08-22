"""Shared helpers for ordering and truncating features across plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from shaply.enums import FeatureOrdering

if TYPE_CHECKING:
    import numpy.typing as npt

    from shaply.explanation import Explanation

    IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Result of ordering and truncating features for display.

    Attributes
    ----------
    order
        Feature indices in display order (bottom-to-top for horizontal plots).
    labels
        Feature names aligned with ``order``.
    grouped_indices
        Indices folded into a "sum of other features" pseudo-row, empty when
        no truncation happened.
    grouped_label
        Label for the folded pseudo-row, or ``None`` when nothing was folded.
    """

    order: IntArray
    labels: tuple[str, ...]
    grouped_indices: IntArray
    grouped_label: str | None


def resolve_feature_index(explanation: Explanation, feature: str | int) -> int:
    """Resolve a feature reference (name or index) to a positional index.

    Parameters
    ----------
    explanation
        The explanation whose features are being referenced.
    feature
        A feature name or an integer index (negative indices allowed).

    Returns
    -------
    int
        The non-negative positional index of the feature.

    Raises
    ------
    IndexError
        If an integer index is out of range.
    KeyError
        If a feature name is unknown.
    """
    if isinstance(feature, int):
        if not -explanation.n_features <= feature < explanation.n_features:
            msg = f"feature index {feature} out of range"
            raise IndexError(msg)
        return feature % explanation.n_features
    try:
        return explanation.feature_names.index(feature)
    except ValueError:
        msg = f"unknown feature {feature!r}; available: {explanation.feature_names}"
        raise KeyError(msg) from None


def _rank(explanation: Explanation, ordering: FeatureOrdering) -> IntArray:
    """Return feature indices ranked from most to least prominent."""
    if ordering is FeatureOrdering.ORIGINAL:
        return np.arange(explanation.n_features)
    if ordering is FeatureOrdering.ALPHABETICAL:
        return np.argsort(explanation.feature_names)
    if ordering is FeatureOrdering.MAX_ABSOLUTE:
        score = np.abs(explanation.values).max(axis=0)
    else:  # IMPORTANCE
        score = explanation.mean_abs()
    return np.argsort(score)[::-1]


def compute_layout(
    explanation: Explanation,
    ordering: FeatureOrdering,
    max_display: int,
) -> FeatureLayout:
    """Order features and fold the overflow into a single grouped row.

    Parameters
    ----------
    explanation
        The explanation whose features are being displayed.
    ordering
        Strategy used to rank features.
    max_display
        Maximum number of individual features to keep.

    Returns
    -------
    FeatureLayout
        The display order, labels and any grouped overflow.
    """
    ranked = _rank(explanation, ordering)
    names = explanation.feature_names

    if explanation.n_features <= max_display:
        top = ranked
        grouped = np.array([], dtype=np.intp)
        grouped_label = None
    else:
        keep = max_display - 1
        top = ranked[:keep]
        grouped = ranked[keep:]
        grouped_label = f"Sum of {grouped.size} other features"

    labels = tuple(names[i] for i in top)
    return FeatureLayout(
        order=top,
        labels=labels,
        grouped_indices=grouped,
        grouped_label=grouped_label,
    )
