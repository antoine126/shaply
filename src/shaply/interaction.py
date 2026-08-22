"""Internal model for SHAP interaction values and its coercion helper.

SHAP interaction values form a ``(n_samples, n_features, n_features)`` tensor
(e.g. from ``shap.TreeExplainer(model).shap_interaction_values(X)``): entry
``[k, i, j]`` is how much the *joint* presence of features ``i`` and ``j``
contributes to sample ``k``'s prediction. The diagonal holds the main effects,
off-diagonal entries the pairwise interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InteractionValues:
    """Normalized SHAP interaction tensor used by the interaction heatmap.

    Attributes
    ----------
    values
        Interaction values, shape ``(n_samples, n_features, n_features)``.
    feature_names
        Feature labels, length ``n_features``.
    """

    values: FloatArray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the tensor is a stack of square interaction matrices."""
        if self.values.ndim != 3 or self.values.shape[1] != self.values.shape[2]:
            msg = (
                "interaction values must have shape (n_samples, n_features, n_features), "
                f"got {self.values.shape}"
            )
            raise ValueError(msg)
        if len(self.feature_names) != self.values.shape[1]:
            msg = (
                f"feature_names must have length {self.values.shape[1]}, "
                f"got {len(self.feature_names)}"
            )
            raise ValueError(msg)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return int(self.values.shape[1])

    def mean_abs_matrix(self) -> FloatArray:
        """Mean absolute interaction per feature pair, shape ``(n_features, n_features)``."""
        return np.abs(self.values).mean(axis=0)


def to_interaction_values(
    values: npt.ArrayLike | object,
    *,
    feature_names: Sequence[str] | None = None,
) -> InteractionValues:
    """Coerce an interaction tensor (or ``.values``-carrying object) into a model.

    Parameters
    ----------
    values
        A ``(n_samples, n_features, n_features)`` array, or an object exposing
        such an array as ``.values`` (and optionally ``.feature_names``).
    feature_names
        Feature labels; inferred from the object or generated when omitted.

    Returns
    -------
    InteractionValues
        The normalized internal representation.
    """
    raw = getattr(values, "values", values)
    array = np.asarray(raw, dtype=np.float64)

    if feature_names is None:
        obj_names = getattr(values, "feature_names", None)
        feature_names = None if obj_names is None else [str(name) for name in obj_names]

    n_features = array.shape[1] if array.ndim == 3 else 0
    resolved = (
        tuple(f"Feature {i}" for i in range(n_features))
        if feature_names is None
        else tuple(str(name) for name in feature_names)
    )
    return InteractionValues(values=array, feature_names=resolved)
