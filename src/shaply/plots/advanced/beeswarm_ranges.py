"""Combined beeswarm + feature-value-ranges figure.

The left panel is the usual SHAP beeswarm (impact on the model output); the right
panel shows, on a shared feature axis, the **real** distribution of each feature's
values as a horizontal violin with an inner box. Because raw features live on very
different scales, each violin is min-max normalized for geometry while the true
``min``/``max`` are annotated at its ends - so an engineer reads both the impact
and the concrete operating range on the same line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shaply.colors import SHAP_BLUE, SHAP_GRAY, SHAP_RED
from shaply.config import BeeswarmRangesConfig
from shaply.explanation import to_explanation
from shaply.plots._common.ordering import compute_layout
from shaply.plots.usual.beeswarm import beeswarm_scatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.intp]


def beeswarm_ranges(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: BeeswarmRangesConfig | None = None,
) -> go.Figure:
    """Render a beeswarm alongside each feature's real value distribution.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.BeeswarmRangesConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The two-panel figure.

    Raises
    ------
    ValueError
        If feature values (``data``) were not provided; the right panel needs
        the real feature values.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    if explanation.data is None:
        msg = "beeswarm_ranges requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    cfg = config or BeeswarmRangesConfig()
    return _build(explanation, cfg)


def _normalize_geometry(column: FloatArray) -> tuple[FloatArray, float, float]:
    """Return ``(values_in_0_1, real_min, real_max)`` for one feature column."""
    finite = column[np.isfinite(column)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi <= lo:
        return np.full_like(column, 0.5), lo, hi
    return (column - lo) / (hi - lo), lo, hi


def _add_range_violins(
    fig: go.Figure,
    explanation: Explanation,
    order: IntArray,
    cfg: BeeswarmRangesConfig,
) -> None:
    assert explanation.data is not None  # guaranteed by caller
    for row, feature_idx in enumerate(order):
        raw = explanation.data[:, feature_idx]
        normalized, lo, hi = _normalize_geometry(raw)
        fig.add_trace(
            go.Violin(
                x=normalized,
                y=np.full_like(normalized, row),
                orientation="h",
                width=0.85,
                side="both",
                points=False,
                box_visible=True,
                meanline_visible=True,
                line_color=SHAP_GRAY,
                fillcolor="rgba(119, 119, 119, 0.25)",
                customdata=raw,
                hovertemplate="value %{customdata:.4g}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        if cfg.show_value_labels:
            fig.add_annotation(
                x=0.0,
                y=row,
                xref="x2",
                yref="y2",
                text=f"{lo:.3g}",
                showarrow=False,
                xanchor="right",
                xshift=-4,
                font={"size": 10, "color": SHAP_BLUE},
            )
            fig.add_annotation(
                x=1.0,
                y=row,
                xref="x2",
                yref="y2",
                text=f"{hi:.3g}",
                showarrow=False,
                xanchor="left",
                xshift=4,
                font={"size": 10, "color": SHAP_RED},
            )


def _build(explanation: Explanation, cfg: BeeswarmRangesConfig) -> go.Figure:
    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    order = layout.order[::-1]  # bottom-to-top: least important first
    labels = list(layout.labels[::-1])
    ratio = cfg.impact_panel_ratio

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        column_widths=[ratio, 1.0 - ratio],
        subplot_titles=("SHAP impact", "Feature value range"),
    )

    fig.add_trace(
        beeswarm_scatter(
            explanation,
            order,
            cfg.color_scale,
            point_size=cfg.point_size,
            opacity=cfg.opacity,
            jitter=cfg.jitter,
        ),
        row=1,
        col=1,
    )
    _add_range_violins(fig, explanation, order, cfg)

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        showgrid=False,
        row=1,
        col=1,
    )
    # Right panel x-axis is a per-feature normalized position: hide its ticks,
    # the real numbers are shown as annotations instead.
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[-0.15, 1.15], row=1, col=2)
    fig.update_xaxes(
        title_text="SHAP value (impact on model output)", showgrid=cfg.show_grid, row=1, col=1
    )
    fig.update_layout(
        title=cfg.title or "SHAP impact and feature value ranges",
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
    )
    fig.add_vline(x=0.0, line_color=SHAP_GRAY, line_width=1, opacity=0.5, row=1, col=1)
    return fig
