"""Combined beeswarm + feature-value-ranges figure.

The left panel is the usual SHAP beeswarm (impact on the model output); the right
panel shows, on a shared feature axis, the **real** distribution of each feature's
values as a horizontal density silhouette with an inner box. The silhouette is
colored with the same low-to-high scale as the beeswarm dots, so a glance at the
color already says which end is "low" and which is "high". Because raw features
live on very different scales, each silhouette is min-max normalized for geometry
while the true ``min``/``max`` are annotated at its ends - so an engineer reads
the impact, the operating range, and the value scale on the same line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shaply.colors import SHAP_BLUE, SHAP_GRAY, SHAP_RED, sample_scale
from shaply.config import BeeswarmRangesConfig
from shaply.explanation import to_explanation
from shaply.plots._common.ordering import compute_layout
from shaply.plots.usual.beeswarm import beeswarm_scatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.enums import ColorScale
    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.intp]

#: Grid resolution and geometry of the gradient-filled density silhouette.
#: 30 color bands per silhouette already reads as a smooth gradient while
#: keeping the trace count per row (bands + outline + box) reasonable.
_GRID_POINTS = 31
_MAX_HALF_WIDTH = 0.42
_BOX_HALF_HEIGHT = 0.11
_BOX_LINE_COLOR = "#333333"


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


def _kde_density(normalized: FloatArray, grid: FloatArray) -> FloatArray:
    """Gaussian KDE of already-``[0, 1]``-normalized values, evaluated on ``grid``."""
    n = normalized.size
    std = float(normalized.std())
    if n < 2 or std == 0.0:
        return np.zeros_like(grid)
    bandwidth = max(1.06 * std * n ** (-1 / 5), 1e-3)  # Silverman's rule of thumb
    diffs = (grid[:, None] - normalized[None, :]) / bandwidth
    density: FloatArray = np.exp(-0.5 * diffs**2).sum(axis=1)
    density /= n * bandwidth * np.sqrt(2 * np.pi)
    return density


def _silhouette_traces(
    normalized: FloatArray, row: int, color_scale: ColorScale
) -> list[go.Scatter]:
    """Gradient-filled density silhouette (many colored bands + a thin outline)."""
    grid: FloatArray = np.linspace(0.0, 1.0, _GRID_POINTS)
    density = _kde_density(normalized, grid)
    peak = float(density.max())
    half_width = density / peak * _MAX_HALF_WIDTH if peak > 0 else np.zeros_like(density)
    top = row + half_width
    bottom = row - half_width

    midpoints = ((grid[:-1] + grid[1:]) / 2).tolist()
    band_colors = sample_scale(color_scale, midpoints)

    traces = [
        go.Scatter(
            x=[grid[i], grid[i + 1], grid[i + 1], grid[i]],
            y=[bottom[i], bottom[i + 1], top[i + 1], top[i]],
            mode="lines",
            fill="toself",
            fillcolor=band_colors[i],
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
        for i in range(_GRID_POINTS - 1)
    ]
    traces.append(
        go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([top, bottom[::-1]]),
            mode="lines",
            line={"color": SHAP_GRAY, "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return traces


def _box_trace(normalized: FloatArray, row: int) -> go.Scatter:
    """Quartile box + whiskers + median/mean ticks, drawn over the silhouette."""
    q1, median, q3 = np.percentile(normalized, [25, 50, 75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    within = normalized[(normalized >= lower_fence) & (normalized <= upper_fence)]
    whisker_lo = float(within.min()) if within.size else float(normalized.min())
    whisker_hi = float(within.max()) if within.size else float(normalized.max())
    mean = float(normalized.mean())
    half = _BOX_HALF_HEIGHT

    # A single polyline: whisker, box outline, median tick, gap, mean tick.
    x = [whisker_lo, q1, None, q1, q3, q3, q1, q1, None, q3, whisker_hi, None]
    y = [row, row, None, row - half, row - half, row + half, row + half, row - half, None]
    y += [row, row, None]
    x += [median, median, None, mean, mean]
    y += [row - half, row + half, None, row - half * 0.7, row + half * 0.7]
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": _BOX_LINE_COLOR, "width": 1.3},
        hoverinfo="skip",
        showlegend=False,
    )


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
        for trace in _silhouette_traces(normalized, row, cfg.color_scale):
            fig.add_trace(trace, row=1, col=2)
        fig.add_trace(_box_trace(normalized, row), row=1, col=2)
        if cfg.show_value_labels:
            fig.add_annotation(
                x=-0.05,
                y=row,
                xref="x2",
                yref="y2",
                text=f"{lo:.3g}",
                showarrow=False,
                xanchor="right",
                font={"size": 10, "color": SHAP_BLUE},
            )
            fig.add_annotation(
                x=1.05,
                y=row,
                xref="x2",
                yref="y2",
                text=f"{hi:.3g}",
                showarrow=False,
                xanchor="left",
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
