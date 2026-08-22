"""Heatmap plot of SHAP values across instances (``shap.plots.heatmap``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import resolve_colorscale
from shaply.config import HeatmapConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


def heatmap(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: HeatmapConfig | None = None,
) -> go.Figure:
    """Render a heatmap of SHAP values with instances on the x-axis.

    Rows are features (ordered by importance), columns are instances (ordered by
    their total SHAP output so similar explanations sit together).

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.HeatmapConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap figure.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or HeatmapConfig()
    return _build_heatmap(explanation, cfg)


def _build_heatmap(explanation: Explanation, cfg: HeatmapConfig) -> go.Figure:
    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    order = layout.order
    labels = list(layout.labels)

    matrix = explanation.values[:, order].T  # (n_display_features, n_samples)

    if layout.grouped_label is not None:
        grouped = explanation.values[:, layout.grouped_indices].sum(axis=1, keepdims=True)
        matrix = np.vstack([matrix, grouped.T])
        labels.append(layout.grouped_label)

    # Order instances by their aggregate output for a readable structure.
    instance_order = np.argsort(explanation.values.sum(axis=1))
    matrix = matrix[:, instance_order]

    # Reverse rows so the most important feature is at the top.
    matrix = matrix[::-1]
    labels.reverse()

    bound = float(np.abs(matrix).max()) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            y=labels,
            colorscale=resolve_colorscale(cfg.color_scale),
            zmid=0.0,
            zmin=-bound,
            zmax=bound,
            colorbar={"title": {"text": "SHAP value", "side": "right"}},
            hovertemplate="instance %{x}<br>%{y}<br>SHAP %{z:.4f}<extra></extra>",
        )
    )
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "SHAP heatmap",
        xaxis_title="Instances",
        yaxis_title=None,
    )
    fig.update_xaxes(showticklabels=False)
    return fig
