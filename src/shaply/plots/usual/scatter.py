"""Dependence / scatter plot (``shap.plots.scatter`` equivalent)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from shaply.colors import SHAP_RED, resolve_colorscale
from shaply.config import ScatterConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import resolve_feature_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


def scatter(
    values: ExplanationLike | npt.ArrayLike | object,
    feature: str | int,
    *,
    color_feature: str | int | None = None,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ScatterConfig | None = None,
) -> go.Figure:
    """Render a SHAP dependence plot for a single feature.

    The x-axis is the feature's value, the y-axis its SHAP value. Points may be
    colored by a second (interaction) feature.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    feature
        Name or index of the feature to place on the axes.
    color_feature
        Optional name or index of an interaction feature used for coloring.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ScatterConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The dependence figure.

    Raises
    ------
    ValueError
        If feature values (``data``) were not provided.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    if explanation.data is None:
        msg = "scatter requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    cfg = config or ScatterConfig()
    return _build_scatter(explanation, cfg, feature, color_feature)


def _build_scatter(
    explanation: Explanation,
    cfg: ScatterConfig,
    feature: str | int,
    color_feature: str | int | None,
) -> go.Figure:
    assert explanation.data is not None  # guaranteed by caller
    idx = resolve_feature_index(explanation, feature)
    x = explanation.data[:, idx]
    y = explanation.values[:, idx]

    marker: dict[str, object] = {"size": cfg.point_size, "opacity": cfg.opacity}
    if color_feature is not None:
        color_idx = resolve_feature_index(explanation, color_feature)
        marker.update(
            {
                "color": explanation.data[:, color_idx],
                "colorscale": resolve_colorscale(cfg.color_scale),
                "colorbar": {
                    "title": {
                        "text": explanation.feature_names[color_idx],
                        "side": "right",
                    }
                },
            }
        )
    else:
        marker["color"] = SHAP_RED

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            hovertemplate="value %{x:.4g}<br>SHAP %{y:.4f}<extra></extra>",
        )
    )
    name = explanation.feature_names[idx]
    apply_layout(
        fig,
        cfg,
        title=cfg.title or f"SHAP dependence - {name}",
        xaxis_title=name,
        yaxis_title=f"SHAP value for {name}",
    )
    return fig
