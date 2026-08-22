"""Pairwise SHAP-interaction heatmap.

Given a SHAP interaction tensor, this ranks features by total interaction
strength and shows the matrix of mean absolute interaction between the top pairs.
Bright off-diagonal cells flag features that act *together* on the prediction -
the closest SHAP gets to surfacing coupled effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import resolve_colorscale
from shaply.config import InteractionHeatmapConfig
from shaply.interaction import to_interaction_values

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.interaction import InteractionValues


def interaction_heatmap(
    values: npt.ArrayLike | object,
    *,
    feature_names: Sequence[str] | None = None,
    config: InteractionHeatmapConfig | None = None,
) -> go.Figure:
    """Render a heatmap of mean absolute SHAP interactions between features.

    Parameters
    ----------
    values
        A ``(n_samples, n_features, n_features)`` interaction tensor (e.g. from
        ``shap.TreeExplainer(model).shap_interaction_values(X)``) or an object
        exposing it as ``.values``.
    feature_names
        Feature labels; inferred or generated when omitted.
    config
        Optional :class:`~shaply.config.InteractionHeatmapConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The interaction heatmap.
    """
    interactions = to_interaction_values(values, feature_names=feature_names)
    cfg = config or InteractionHeatmapConfig()
    return _build(interactions, cfg)


def _build(interactions: InteractionValues, cfg: InteractionHeatmapConfig) -> go.Figure:
    matrix = interactions.mean_abs_matrix()
    names = interactions.feature_names

    # Rank features by total interaction strength (off-diagonal), keep the top-k.
    strength = matrix.sum(axis=1) - np.diag(matrix)
    order = np.argsort(strength)[::-1][: cfg.max_display]

    sub = matrix[np.ix_(order, order)].copy()
    labels = [names[i] for i in order]
    if not cfg.show_diagonal:
        np.fill_diagonal(sub, np.nan)

    fig = go.Figure(
        go.Heatmap(
            z=sub,
            x=labels,
            y=labels,
            colorscale=resolve_colorscale(cfg.color_scale),
            colorbar={"title": {"text": "mean|interaction|", "side": "right"}},
            text=np.round(sub, 3) if cfg.show_values else None,
            texttemplate="%{text}" if cfg.show_values else None,
            hovertemplate="%{y} × %{x}<br>mean|interaction| %{z:.4f}<extra></extra>",  # noqa: RUF001
        )
    )
    fig.update_layout(
        title=cfg.title or "SHAP interaction strength",
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    fig.update_xaxes(showgrid=False)
    # Display the strongest feature at the top while keeping the data (and its
    # diagonal) aligned between both axes.
    fig.update_yaxes(showgrid=False, autorange="reversed", scaleanchor="x", constrain="domain")
    return fig
