"""Error-analysis plot: which features drive the model's mistakes.

Splits the instances into a *correct* and an *error* cohort and compares the mean
SHAP value of each feature between them, ordered by the size of the gap. Features
with a large gap are the ones the model relies on differently when it is wrong -
usually the most profitable place to look in an industrial setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_BLUE, SHAP_GRAY, SHAP_RED
from shaply.config import ErrorAnalysisConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    BoolArray = npt.NDArray[np.bool_]


def error_analysis(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    errors: npt.ArrayLike | None = None,
    y_true: npt.ArrayLike | None = None,
    y_pred: npt.ArrayLike | None = None,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ErrorAnalysisConfig | None = None,
) -> go.Figure:
    """Compare mean SHAP values between correct and mis-predicted instances.

    Provide the error cohort either directly via ``errors`` (a boolean mask, True
    for mistakes) or via ``y_true`` and ``y_pred`` (mismatch defines an error).

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    errors
        Boolean mask, ``True`` where the model was wrong.
    y_true, y_pred
        Ground-truth and predicted labels; an error is ``y_true != y_pred``.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ErrorAnalysisConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The error-analysis figure.

    Raises
    ------
    ValueError
        If the error cohort is under-specified, mis-sized, or empty/complete.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    mask = _resolve_mask(explanation.n_samples, errors=errors, y_true=y_true, y_pred=y_pred)
    cfg = config or ErrorAnalysisConfig()
    return _build(explanation, cfg, mask)


def _resolve_mask(
    n_samples: int,
    *,
    errors: npt.ArrayLike | None,
    y_true: npt.ArrayLike | None,
    y_pred: npt.ArrayLike | None,
) -> BoolArray:
    if errors is not None:
        mask = np.asarray(errors, dtype=bool)
    elif y_true is not None and y_pred is not None:
        mask = np.asarray(y_true) != np.asarray(y_pred)
    else:
        msg = "provide either errors=... or both y_true=... and y_pred=..."
        raise ValueError(msg)

    if mask.shape != (n_samples,):
        msg = f"error mask must have shape ({n_samples},), got {mask.shape}"
        raise ValueError(msg)
    if not mask.any() or mask.all():
        msg = "error mask must contain both correct and erroneous instances"
        raise ValueError(msg)
    return mask


def _build(explanation: Explanation, cfg: ErrorAnalysisConfig, mask: BoolArray) -> go.Figure:
    mean_error = explanation.values[mask].mean(axis=0)
    mean_correct = explanation.values[~mask].mean(axis=0)
    gap = np.abs(mean_error - mean_correct)

    order = np.argsort(gap)[::-1][: cfg.max_display]
    # Reverse for display so the largest gap sits at the top.
    order = order[::-1]
    labels = [explanation.feature_names[i] for i in order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(mean_correct[order]),
            y=labels,
            orientation="h",
            name="correct",
            marker_color=SHAP_GRAY,
            text=[f"{v:+.2f}" for v in mean_correct[order]] if cfg.show_values else None,
            textposition="auto",
            hovertemplate="%{y} (correct): %{x:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=list(mean_error[order]),
            y=labels,
            orientation="h",
            name="error",
            marker_color=SHAP_RED,
            text=[f"{v:+.2f}" for v in mean_error[order]] if cfg.show_values else None,
            textposition="auto",
            hovertemplate="%{y} (error): %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.02, "x": 0})
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "Error analysis - mean SHAP by cohort",
        xaxis_title="mean SHAP value",
        yaxis_title=None,
    )
    fig.add_vline(x=0.0, line_color=SHAP_BLUE, line_width=1, opacity=0.4)
    return fig
