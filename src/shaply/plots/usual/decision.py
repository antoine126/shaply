"""Decision plot (``shap.decision_plot`` equivalent).

Each instance is drawn as a line that starts at the model's expected value at
the bottom axis and accumulates SHAP values feature by feature going upward, so
its horizontal position at the top is the prediction ``f(x)``. Lines are colored
by their predicted output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_GRAY, sample_scale
from shaply.config import DecisionConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]


def decision(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: DecisionConfig | None = None,
) -> go.Figure:
    """Render a decision plot as a Plotly figure.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.DecisionConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The decision figure.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or DecisionConfig()
    return _build_decision(explanation, cfg)


def _ordered_contributions(
    explanation: Explanation,
    cfg: DecisionConfig,
) -> tuple[FloatArray, list[str]]:
    """Return the ``(n_samples, k)`` contribution matrix and labels, bottom-up.

    Features are ordered least-important first (bottom of the plot); any overflow
    is folded into a single grouped column placed at the very bottom.
    """
    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    bottom_up = layout.order[::-1]
    matrix = explanation.values[:, bottom_up]
    labels = [explanation.feature_names[i] for i in bottom_up]

    if layout.grouped_label is not None:
        grouped = explanation.values[:, layout.grouped_indices].sum(axis=1, keepdims=True)
        matrix = np.hstack([grouped, matrix])
        labels = [layout.grouped_label, *labels]
    return matrix, labels


def _build_decision(explanation: Explanation, cfg: DecisionConfig) -> go.Figure:
    matrix, labels = _ordered_contributions(explanation, cfg)
    base = explanation.base_values
    n_samples, n_display = matrix.shape

    # Path x-coordinates: base, then cumulative sum up through each feature.
    cumulative = base[:, None] + np.cumsum(matrix, axis=1)
    x_paths = np.hstack([base[:, None], cumulative])  # (n_samples, n_display + 1)
    y_levels = list(range(-1, n_display))  # -1 is the base axis, 0..k-1 the features.

    predictions = x_paths[:, -1]
    lo, hi = float(predictions.min()), float(predictions.max())
    span = hi - lo or 1.0
    fractions = [(float(p) - lo) / span for p in predictions]
    colors = sample_scale(cfg.color_scale, fractions)

    fig = go.Figure()
    for i in range(n_samples):
        fig.add_trace(
            go.Scatter(
                x=x_paths[i],
                y=y_levels,
                mode="lines",
                line={"color": colors[i], "width": cfg.line_width},
                opacity=cfg.opacity,
                showlegend=False,
                hovertemplate=f"instance {i}<br>output %{{x:.4f}}<extra></extra>",
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_display)),
        ticktext=labels,
        showgrid=False,
    )
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "Decision plot",
        xaxis_title="Model output",
        yaxis_title=None,
    )
    fig.add_vline(
        x=float(np.mean(base)),
        line_color=SHAP_GRAY,
        line_width=1,
        line_dash="dot",
        annotation_text="base",
        annotation_position="top",
    )
    return fig
