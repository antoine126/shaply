"""Feature-clustering / redundancy heatmap based on SHAP similarity.

Two features are *redundant* for the model when their per-sample SHAP vectors are
strongly correlated: they push predictions in lockstep. This plot shows the
SHAP-correlation matrix reordered by average-linkage clustering, so blocks of
mutually correlated (often droppable) features stand out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import resolve_colorscale
from shaply.config import FeatureClusteringConfig
from shaply.explanation import to_explanation
from shaply.plots._common.cluster import average_linkage_order

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]


def feature_clustering(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: FeatureClusteringConfig | None = None,
) -> go.Figure:
    """Render a clustered heatmap of SHAP correlation between features.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.FeatureClusteringConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The clustered correlation heatmap.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or FeatureClusteringConfig()
    return _build(explanation, cfg)


def _shap_correlation(values: FloatArray) -> FloatArray:
    """Pearson correlation between the SHAP vectors of each feature pair."""
    std = values.std(axis=0)
    safe = np.where(std == 0.0, 1.0, std)
    normalized = (values - values.mean(axis=0)) / safe
    corr = (normalized.T @ normalized) / values.shape[0]
    # Features with no SHAP variance are uncorrelated with everything.
    dead = std == 0.0
    corr[dead, :] = 0.0
    corr[:, dead] = 0.0
    np.fill_diagonal(corr, 1.0)
    clipped: FloatArray = np.clip(corr, -1.0, 1.0)
    return clipped


def _build(explanation: Explanation, cfg: FeatureClusteringConfig) -> go.Figure:
    corr = _shap_correlation(explanation.values)
    # Distance folds sign: strong positive OR negative correlation => redundant.
    distance = 1.0 - np.abs(corr)
    order = average_linkage_order(distance)

    ordered = corr[np.ix_(order, order)]
    labels = [explanation.feature_names[i] for i in order]

    fig = go.Figure(
        go.Heatmap(
            z=ordered,
            x=labels,
            y=labels,
            colorscale=resolve_colorscale(cfg.color_scale),
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": {"text": "SHAP corr", "side": "right"}},
            text=np.round(ordered, 2) if cfg.show_values else None,
            texttemplate="%{text}" if cfg.show_values else None,
            hovertemplate="%{y} / %{x}<br>corr %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=cfg.title or "Feature redundancy (SHAP correlation, clustered)",
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed", scaleanchor="x", constrain="domain")
    return fig
