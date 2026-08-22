"""Waterfall plot for a single prediction (``shap.plots.waterfall`` equivalent)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from shaply.colors import SHAP_BLUE, SHAP_GRAY, SHAP_RED
from shaply.config import WaterfallConfig
from shaply.enums import FeatureOrdering
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike


def waterfall(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    sample_index: int = 0,
    config: WaterfallConfig | None = None,
) -> go.Figure:
    """Render a waterfall plot explaining a single prediction.

    The plot starts at the model's expected value ``E[f(x)]`` and adds each
    feature's contribution to reach the prediction ``f(x)``.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    sample_index
        Index of the instance to explain when the input holds several samples.
    config
        Optional :class:`~shaply.config.WaterfallConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The waterfall figure.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    single = explanation.select_sample(sample_index)
    cfg = config or WaterfallConfig()
    return _build_waterfall(single, cfg)


def _feature_label(explanation: Explanation, idx: int) -> str:
    name = explanation.feature_names[idx]
    if explanation.data is None:
        return name
    return f"{name} = {explanation.data[0, idx]:.3g}"


def _build_waterfall(explanation: Explanation, cfg: WaterfallConfig) -> go.Figure:
    contributions = explanation.values[0]
    base = float(explanation.base_values[0])
    prediction = base + float(contributions.sum())

    layout = compute_layout(explanation, FeatureOrdering.IMPORTANCE, cfg.max_display)

    steps: list[float] = [float(contributions[i]) for i in layout.order]
    labels: list[str] = [_feature_label(explanation, int(i)) for i in layout.order]

    if layout.grouped_label is not None:
        steps.append(float(contributions[layout.grouped_indices].sum()))
        labels.append(layout.grouped_label)

    # Smallest contribution near the base, largest near the prediction (top).
    steps.reverse()
    labels.reverse()

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=labels,
            x=steps,
            base=base,
            measure=["relative"] * len(steps),
            text=[f"{v:+.2f}" for v in steps] if cfg.show_values else None,
            textposition="outside",
            connector={"line": {"color": SHAP_GRAY, "width": 1}},
            increasing={"marker": {"color": SHAP_RED}},
            decreasing={"marker": {"color": SHAP_BLUE}},
            hovertemplate="%{y}: %{delta:+.4f}<extra></extra>",
        )
    )
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "Waterfall explanation",
        xaxis_title="Model output",
        yaxis_title=None,
    )
    fig.add_vline(
        x=base,
        line_color=SHAP_GRAY,
        line_width=1,
        line_dash="dot",
        annotation_text=f"E[f(x)] = {base:.2f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=prediction,
        line_color=SHAP_GRAY,
        line_width=1,
        annotation_text=f"f(x) = {prediction:.2f}",
        annotation_position="bottom",
    )
    return fig
