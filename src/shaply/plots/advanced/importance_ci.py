"""Global importance with bootstrap confidence intervals.

Like the bar plot, but each ``mean(|SHAP|)`` bar carries a bootstrap confidence
interval over instances, so a fragile ranking (wide, overlapping intervals) is
not mistaken for a robust one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_RED
from shaply.config import ImportanceCIConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]


def importance_ci(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ImportanceCIConfig | None = None,
) -> go.Figure:
    """Render global feature importance with bootstrap confidence intervals.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ImportanceCIConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The importance figure with error bars.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or ImportanceCIConfig()
    return _build(explanation, cfg)


def _bootstrap_ci(abs_values: FloatArray, cfg: ImportanceCIConfig) -> tuple[FloatArray, FloatArray]:
    """Return per-feature ``(low, high)`` bounds of mean(|SHAP|) over resamples."""
    rng = np.random.default_rng(cfg.random_state)
    n_samples = abs_values.shape[0]
    means = np.empty((cfg.n_boot, abs_values.shape[1]), dtype=np.float64)
    for b in range(cfg.n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        means[b] = abs_values[idx].mean(axis=0)
    alpha = (1.0 - cfg.ci) / 2.0
    low = np.quantile(means, alpha, axis=0)
    high = np.quantile(means, 1.0 - alpha, axis=0)
    return low, high


def _build(explanation: Explanation, cfg: ImportanceCIConfig) -> go.Figure:
    abs_values = np.abs(explanation.values)
    importance = abs_values.mean(axis=0)
    low, high = _bootstrap_ci(abs_values, cfg)

    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    order = layout.order[::-1]  # least important at the bottom
    labels = list(layout.labels[::-1])

    values_ordered = importance[order]
    err_plus = high[order] - values_ordered
    err_minus = values_ordered - low[order]

    fig = go.Figure(
        go.Bar(
            x=values_ordered,
            y=labels,
            orientation="h",
            marker_color=SHAP_RED,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "color": "#444444",
                "thickness": 1.2,
            },
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    apply_layout(
        fig,
        cfg,
        title=cfg.title or f"Global importance with {int(cfg.ci * 100)}% CI",
        xaxis_title="mean(|SHAP value|)",
        yaxis_title=None,
    )
    return fig
