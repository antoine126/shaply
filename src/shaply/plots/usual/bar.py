"""Global feature-importance bar plot (``shap.plots.bar`` equivalent)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_BLUE, SHAP_RED, signed_color
from shaply.config import BarConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


def bar(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: BarConfig | None = None,
) -> go.Figure:
    """Render a feature-importance bar plot as a Plotly figure.

    For a multi-sample explanation, bars encode the mean absolute SHAP value
    per feature. For a single instance, bars encode the signed contributions
    and are colored red (positive) or blue (negative), matching ``shap``.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.BarConfig`; defaults are used otherwise.

    Returns
    -------
    plotly.graph_objects.Figure
        The bar figure.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or BarConfig()
    return _build_bar(explanation, cfg)


def _build_bar(explanation: Explanation, cfg: BarConfig) -> go.Figure:
    signed = explanation.is_single
    magnitudes = explanation.values[0] if signed else explanation.mean_abs()

    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    bar_values = list(magnitudes[layout.order])
    labels = list(layout.labels)

    if layout.grouped_label is not None:
        grouped_value = magnitudes[layout.grouped_indices]
        bar_values.append(float(np.sum(np.abs(grouped_value) if not signed else grouped_value)))
        labels.append(layout.grouped_label)

    # Plotly draws the first category at the bottom; reverse so the most
    # important feature sits at the top like in shap.
    bar_values.reverse()
    labels.reverse()

    colors = [signed_color(v) for v in bar_values] if signed else [SHAP_RED] * len(bar_values)

    fig = go.Figure(
        go.Bar(
            x=bar_values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" if signed else f"{v:.2f}" for v in bar_values]
            if cfg.show_values
            else None,
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    x_title = "SHAP value" if signed else "mean(|SHAP value|)"
    default_title = cfg.title or (
        "Feature contributions" if signed else "Global feature importance"
    )
    apply_layout(
        fig,
        cfg,
        title=default_title,
        xaxis_title=x_title,
        yaxis_title=None,
    )
    if signed:
        fig.add_vline(x=0.0, line_color=SHAP_BLUE, line_width=1, opacity=0.4)
    return fig
