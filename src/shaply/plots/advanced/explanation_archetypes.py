"""Explanation archetypes: typical SHAP-profile patterns across instances.

Clusters instances by their SHAP profile and shows each cluster's mean SHAP per
feature as a heatmap row. The archetypes read as the model's recurring "reasons"
- e.g. distinct failure modes or operating regimes - rather than one instance at
a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import resolve_colorscale
from shaply.config import ExplanationArchetypesConfig
from shaply.explanation import to_explanation
from shaply.plots._common.cluster import kmeans
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


def explanation_archetypes(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ExplanationArchetypesConfig | None = None,
) -> go.Figure:
    """Cluster instances by SHAP profile and show each archetype's mean SHAP.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ExplanationArchetypesConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The archetypes heatmap.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or ExplanationArchetypesConfig()
    return _build(explanation, cfg)


def _build(explanation: Explanation, cfg: ExplanationArchetypesConfig) -> go.Figure:
    labels, _ = kmeans(explanation.values, cfg.n_clusters, random_state=cfg.random_state)
    n_clusters = int(labels.max()) + 1

    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    order = layout.order
    feature_labels = list(layout.labels)

    # Mean SHAP profile per cluster, sorted by cluster size (largest first).
    sizes = np.array([int((labels == c).sum()) for c in range(n_clusters)])
    cluster_order = np.argsort(sizes)[::-1]

    rows = []
    row_labels = []
    for c in cluster_order:
        profile = explanation.values[labels == c][:, order].mean(axis=0)
        rows.append(profile)
        share = 100.0 * sizes[c] / explanation.n_samples
        row_labels.append(f"Archetype {c} (n={sizes[c]}, {share:.0f}%)")

    matrix = np.array(rows)
    bound = float(np.abs(matrix).max()) or 1.0

    # Reverse so the largest archetype sits at the top.
    matrix = matrix[::-1]
    row_labels = row_labels[::-1]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=feature_labels,
            y=row_labels,
            colorscale=resolve_colorscale(cfg.color_scale),
            zmid=0.0,
            zmin=-bound,
            zmax=bound,
            colorbar={"title": {"text": "mean SHAP", "side": "right"}},
            hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=cfg.title or f"Explanation archetypes ({n_clusters} SHAP-profile clusters)",
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        xaxis_title="Feature",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
