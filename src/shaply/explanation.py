"""Internal explanation model and coercion helpers.

``shaply`` accepts several input shapes without ever importing ``shap``:

* a ``shap.Explanation``-like object (duck-typed on ``values`` / ``base_values``
  / ``data`` / ``feature_names``),
* raw :class:`numpy.ndarray` SHAP values (optionally with feature values),
* a :class:`pandas.DataFrame` of SHAP values (columns become feature names).

They are all normalized into the frozen :class:`Explanation` dataclass, the
single internal representation every plot builder consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class ExplanationLike(Protocol):
    """Structural type matching ``shap.Explanation`` without importing ``shap``."""

    values: Any
    base_values: Any
    data: Any
    feature_names: Any


@dataclass(frozen=True, slots=True)
class Explanation:
    """Normalized SHAP explanation used internally by every plot builder.

    All arrays are stored two-dimensionally as ``(n_samples, n_features)`` so
    plot code never has to special-case single instances.

    Attributes
    ----------
    values
        SHAP values, shape ``(n_samples, n_features)``.
    base_values
        Model expected value per sample, shape ``(n_samples,)``.
    feature_names
        Feature labels, length ``n_features``.
    data
        Original feature values aligned with ``values``, or ``None`` when the
        caller only provided SHAP values.
    """

    values: FloatArray
    base_values: FloatArray
    feature_names: tuple[str, ...]
    data: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate array shapes are mutually consistent."""
        if self.values.ndim != 2:
            msg = f"values must be 2D (n_samples, n_features), got shape {self.values.shape}"
            raise ValueError(msg)
        n_samples, n_features = self.values.shape
        if self.base_values.shape != (n_samples,):
            msg = f"base_values must have shape ({n_samples},), got {self.base_values.shape}"
            raise ValueError(msg)
        if len(self.feature_names) != n_features:
            msg = f"feature_names must have length {n_features}, got {len(self.feature_names)}"
            raise ValueError(msg)
        if self.data is not None and self.data.shape != self.values.shape:
            msg = f"data must match values shape {self.values.shape}, got {self.data.shape}"
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of explained instances."""
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features."""
        return int(self.values.shape[1])

    @property
    def is_single(self) -> bool:
        """Whether the explanation describes a single instance."""
        return self.n_samples == 1

    def mean_abs(self) -> FloatArray:
        """Mean absolute SHAP value per feature (global importance)."""
        return np.abs(self.values).mean(axis=0)

    def select_sample(self, index: int) -> Explanation:
        """Return a single-sample explanation for ``index``."""
        return Explanation(
            values=self.values[index : index + 1],
            base_values=self.base_values[index : index + 1],
            feature_names=self.feature_names,
            data=None if self.data is None else self.data[index : index + 1],
        )


def _default_feature_names(n_features: int) -> tuple[str, ...]:
    return tuple(f"Feature {i}" for i in range(n_features))


def _as_2d(array: FloatArray) -> FloatArray:
    """Promote a 1D array of SHAP values to ``(1, n_features)``."""
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _coerce_base_values(
    base_values: object,
    n_samples: int,
) -> FloatArray:
    if base_values is None:
        return np.zeros(n_samples, dtype=np.float64)
    arr = np.asarray(base_values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_samples, float(arr), dtype=np.float64)
    if arr.shape == (n_samples,):
        return arr
    msg = f"base_values must be scalar or shape ({n_samples},), got {arr.shape}"
    raise ValueError(msg)


def _from_explanation_like(
    obj: ExplanationLike,
    *,
    output_index: int | None,
) -> Explanation:
    raw_values = np.asarray(obj.values, dtype=np.float64)
    is_multi_output = raw_values.ndim == 3
    values = _as_2d(_select_output(raw_values, output_index))
    n_samples, n_features = values.shape

    data = None if obj.data is None else _as_2d(np.asarray(obj.data, dtype=np.float64))

    raw_names = obj.feature_names
    feature_names = (
        _default_feature_names(n_features)
        if raw_names is None
        else tuple(str(name) for name in raw_names)
    )

    base = _coerce_base_values(
        _select_base_output(obj.base_values, output_index, multi_output=is_multi_output),
        n_samples,
    )
    return Explanation(
        values=values,
        base_values=base,
        feature_names=feature_names,
        data=data,
    )


def _select_base_output(
    base_values: object,
    output_index: int | None,
    *,
    multi_output: bool,
) -> FloatArray | None:
    """Select the class axis of ``base_values`` for a multi-output explanation.

    For a multi-output ``shap.Explanation`` the SHAP values are 3D
    ``(n_samples, n_features, n_outputs)`` while ``base_values`` are 2D
    ``(n_samples, n_outputs)``; the trailing output axis must be indexed too.
    """
    if base_values is None:
        return None
    arr = np.asarray(base_values, dtype=np.float64)
    if multi_output and arr.ndim == 2:
        if output_index is None:
            msg = "Multi-output base_values detected; pass output_index=... to select a class."
            raise ValueError(msg)
        return arr[:, output_index]
    return arr


def _select_output(array: FloatArray, output_index: int | None) -> FloatArray:
    """Select one model output from a multi-output SHAP array.

    Multi-class SHAP values have a trailing output axis. When present, an
    ``output_index`` must be supplied to disambiguate which class to plot.
    """
    if array.ndim <= 2:
        return array
    if array.ndim == 3:
        if output_index is None:
            msg = "Multi-output SHAP values detected (3D). Pass output_index=... to select a class."
            raise ValueError(msg)
        return array[..., output_index]
    msg = f"Unsupported SHAP values with {array.ndim} dimensions"
    raise ValueError(msg)


def to_explanation(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
) -> Explanation:
    """Coerce supported inputs into an :class:`Explanation`.

    Parameters
    ----------
    values
        A ``shap.Explanation``-like object, a numpy array of SHAP values, or a
        :class:`pandas.DataFrame` whose columns are feature names.
    base_values
        Model expected value(s). Scalar or per-sample. Defaults to zeros.
        Ignored when ``values`` already carries base values.
    data
        Original feature values aligned with the SHAP values.
    feature_names
        Feature labels. Inferred from a DataFrame or generated when omitted.
    output_index
        Class index to select when ``values`` holds multi-output SHAP values.

    Returns
    -------
    Explanation
        The normalized internal representation.

    Raises
    ------
    ValueError
        If the provided arrays have inconsistent or unsupported shapes.
    """
    if isinstance(values, ExplanationLike) and not isinstance(values, np.ndarray):
        return _from_explanation_like(values, output_index=output_index)

    frame_names, frame_values = _maybe_dataframe(values)
    if frame_values is not None:
        values_arr = frame_values
        if feature_names is None:
            feature_names = frame_names
    else:
        values_arr = np.asarray(values, dtype=np.float64)

    values_arr = _as_2d(_select_output(values_arr, output_index))
    n_samples, n_features = values_arr.shape

    resolved_names = (
        _default_feature_names(n_features)
        if feature_names is None
        else tuple(str(name) for name in feature_names)
    )

    data_arr = None if data is None else _as_2d(np.asarray(data, dtype=np.float64))
    base_arr = _coerce_base_values(base_values, n_samples)

    return Explanation(
        values=values_arr,
        base_values=base_arr,
        feature_names=resolved_names,
        data=data_arr,
    )


def _maybe_dataframe(obj: object) -> tuple[tuple[str, ...] | None, FloatArray | None]:
    """Extract ``(column_names, values)`` if ``obj`` is a pandas DataFrame."""
    columns = getattr(obj, "columns", None)
    to_numpy = getattr(obj, "to_numpy", None)
    if columns is None or to_numpy is None:
        return None, None
    names = tuple(str(col) for col in columns)
    array = np.asarray(to_numpy(), dtype=np.float64)
    return names, array
