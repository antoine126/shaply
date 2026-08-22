"""Monotonicity check: how cleanly each feature's value drives its SHAP value.

Ranks features by the Spearman rank correlation between their value and their
SHAP value. A magnitude near 1 means a clean monotonic effect (value up →
consistently more/less output); near 0 flags a non-monotonic feature whose effect
is driven by interactions or noise, and is worth a closer look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import signed_color
from shaply.config import MonotonicityConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.stats import spearman

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


def monotonicity_check(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: MonotonicityConfig | None = None,
) -> go.Figure:
    """Rank features by the monotonicity of their value-to-SHAP relationship.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.MonotonicityConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The monotonicity figure.

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
        msg = "monotonicity_check requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    cfg = config or MonotonicityConfig()
    return _build(explanation, cfg)


def _build(explanation: Explanation, cfg: MonotonicityConfig) -> go.Figure:
    assert explanation.data is not None  # guaranteed by caller
    correlations = np.array(
        [
            spearman(explanation.data[:, i], explanation.values[:, i])
            for i in range(explanation.n_features)
        ]
    )

    order = np.argsort(np.abs(correlations))[::-1][: cfg.max_display]
    order = order[::-1]  # largest magnitude at the top
    scores = correlations[order]
    labels = [explanation.feature_names[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=[signed_color(v) for v in scores],
            text=[f"{v:+.2f}" for v in scores],
            textposition="outside",
            hovertemplate="%{y}: Spearman %{x:.3f}<extra></extra>",
        )
    )
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "Monotonicity of feature effects (Spearman value vs SHAP)",
        xaxis_title="Spearman correlation (-1 decreasing ... +1 increasing)",
        yaxis_title=None,
    )
    fig.update_xaxes(range=[-1.05, 1.05])
    fig.add_vline(x=0.0, line_color="#777777", line_width=1, opacity=0.5)
    return fig
