"""Force plot for a single prediction (``shap.plots.force`` equivalent).

Positive contributions (red) push the prediction up and negative ones (blue)
push it down; the two blocks meet at ``f(x)``. The strip therefore spans
``[base + sum(negatives), base + sum(positives)]`` with the red/blue boundary at
the prediction, exactly as in ``shap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from shaply.colors import SHAP_BLUE, SHAP_GRAY, SHAP_RED
from shaply.config import ForceConfig
from shaply.enums import FeatureOrdering
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


@dataclass(frozen=True, slots=True)
class _Segment:
    """One feature block along the force axis."""

    start: float
    width: float
    color: str
    label: str
    value: float


def force(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    sample_index: int = 0,
    config: ForceConfig | None = None,
) -> go.Figure:
    """Render an additive force plot explaining a single prediction.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    sample_index
        Index of the instance to explain when the input holds several samples.
    config
        Optional :class:`~shaply.config.ForceConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The force figure.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    single = explanation.select_sample(sample_index)
    cfg = config or ForceConfig()
    return _build_force(single, cfg)


def _feature_label(explanation: Explanation, idx: int) -> str:
    name = explanation.feature_names[idx]
    if explanation.data is None:
        return name
    return f"{name} = {explanation.data[0, idx]:.3g}"


def _collect_segments(explanation: Explanation, cfg: ForceConfig) -> tuple[_Segment, ...]:
    contributions = explanation.values[0]
    base = float(explanation.base_values[0])
    prediction = base + float(contributions.sum())

    layout = compute_layout(explanation, FeatureOrdering.IMPORTANCE, cfg.max_display)
    pairs: list[tuple[str, float]] = [
        (_feature_label(explanation, int(i)), float(contributions[i])) for i in layout.order
    ]
    if layout.grouped_label is not None:
        pairs.append((layout.grouped_label, float(contributions[layout.grouped_indices].sum())))

    positives = sorted((p for p in pairs if p[1] >= 0), key=lambda p: p[1])
    negatives = sorted((p for p in pairs if p[1] < 0), key=lambda p: -p[1])

    segments: list[_Segment] = []
    # Red block: fills leftwards from f(x); largest positive sits against f(x).
    cursor = prediction - sum(v for _, v in positives)
    for label, value in positives:
        segments.append(_Segment(cursor, value, SHAP_RED, label, value))
        cursor += value
    # Blue block: fills rightwards from f(x); largest negative sits against f(x).
    cursor = prediction
    for label, value in negatives:
        width = -value
        segments.append(_Segment(cursor, width, SHAP_BLUE, label, value))
        cursor += width
    return tuple(segments)


def _build_force(explanation: Explanation, cfg: ForceConfig) -> go.Figure:
    base = float(explanation.base_values[0])
    prediction = base + float(explanation.values[0].sum())
    segments = _collect_segments(explanation, cfg)

    fig = go.Figure(
        go.Bar(
            x=[seg.width for seg in segments],
            base=[seg.start for seg in segments],
            y=[""] * len(segments),
            orientation="h",
            marker_color=[seg.color for seg in segments],
            marker_line_color="white",
            marker_line_width=1,
            text=[f"{seg.value:+.2f}" for seg in segments] if cfg.show_values else None,
            textposition="inside",
            insidetextanchor="middle",
            customdata=[[seg.label, seg.value] for seg in segments],
            hovertemplate="%{customdata[0]}: %{customdata[1]:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(barmode="overlay", bargap=0.0, showlegend=False)
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "Force plot",
        xaxis_title="Model output",
        yaxis_title=None,
    )
    fig.update_yaxes(showticklabels=False)
    fig.add_vline(
        x=base,
        line_color=SHAP_GRAY,
        line_width=1,
        line_dash="dot",
        annotation_text=f"base = {base:.2f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=prediction,
        line_color=SHAP_GRAY,
        line_width=2,
        annotation_text=f"f(x) = {prediction:.2f}",
        annotation_position="top right",
    )
    return fig
