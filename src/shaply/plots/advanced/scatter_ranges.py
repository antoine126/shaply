"""Dependence scatter framed by its two axes' marginal distributions.

The center panel is a standard SHAP dependence plot for one feature: x is the
feature's real value, y its SHAP value, and points are colored by the (also
redundant, but helpful to the eye) feature value on the same low-to-high scale
used across ``shaply``. Around it, four marginal panels repeat each axis'
distribution in two complementary shapes:

- top: a box plot of the real values, gradient-colored left-to-right.
- bottom: a density silhouette of the real values, gradient-colored.
- left: a plain gray density silhouette of the SHAP values.
- right: a plain gray box plot of the SHAP values.

Only the x-axis (real value) marginals are colored, since that color already
carries meaning (low/high); the y-axis (SHAP value) marginals stay neutral.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shaply.colors import SHAP_GRAY, resolve_colorscale
from shaply.config import ScatterRangesConfig
from shaply.explanation import to_explanation
from shaply.plots._common.gradient import gradient_box_trace, gradient_silhouette_traces
from shaply.plots._common.ordering import resolve_feature_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]

#: 3x3 subplot grid: corners are empty, the cross holds the scatter + marginals.
_N_COLS = 3
_TOP_HALF_HEIGHT = 0.4
_BOTTOM_MAX_HALF_WIDTH = 1.0
_GRID_POINTS = 31


def scatter_ranges(
    values: ExplanationLike | npt.ArrayLike | object,
    feature: str | int,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ScatterRangesConfig | None = None,
) -> go.Figure:
    """Render a dependence scatter framed by its axes' marginal distributions.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    feature
        Name or index of the feature to place on the axes.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ScatterRangesConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The framed scatter figure.

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
        msg = "scatter_ranges requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    cfg = config or ScatterRangesConfig()
    return _build(explanation, cfg, feature)


def _axis_ref(row: int, col: int, letter: str) -> str:
    """Return the plotly axis reference (e.g. ``"x5"``) for a subplot cell."""
    index = (row - 1) * _N_COLS + col
    return letter if index == 1 else f"{letter}{index}"


def _build(explanation: Explanation, cfg: ScatterRangesConfig, feature: str | int) -> go.Figure:
    assert explanation.data is not None  # guaranteed by caller
    idx = resolve_feature_index(explanation, feature)
    raw = explanation.data[:, idx]
    shap_vals = explanation.values[:, idx]
    name = explanation.feature_names[idx]

    finite = np.isfinite(raw)
    raw_f, x_lo, x_hi = raw[finite], 0.0, 1.0
    if raw_f.size:
        x_lo, x_hi = float(raw_f.min()), float(raw_f.max())
    normalized = (raw - x_lo) / (x_hi - x_lo) if x_hi > x_lo else np.full_like(raw, 0.5)

    ratio = cfg.marginal_ratio
    fig = make_subplots(
        rows=3,
        cols=3,
        row_heights=[ratio, 1.0 - 2 * ratio, ratio],
        column_widths=[ratio, 1.0 - 2 * ratio, ratio],
        horizontal_spacing=0.015,
        vertical_spacing=0.015,
    )

    fig.add_trace(
        go.Scatter(
            x=raw,
            y=shap_vals,
            mode="markers",
            marker={
                "size": cfg.point_size,
                "opacity": cfg.opacity,
                "color": normalized,
                "colorscale": resolve_colorscale(cfg.color_scale),
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {
                    "title": {"text": "Feature value", "side": "right"},
                    "tickmode": "array",
                    "tickvals": [0.0, 1.0],
                    "ticktext": ["Low", "High"],
                },
            },
            hovertemplate="value %{x:.4g}<br>SHAP %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    for trace in gradient_box_trace(
        raw_f,
        0.0,
        cfg.color_scale,
        half_height=_TOP_HALF_HEIGHT,
        vmin=x_lo,
        vmax=x_hi,
    ):
        fig.add_trace(trace, row=1, col=2)
    for trace in gradient_silhouette_traces(
        raw_f,
        0.0,
        cfg.color_scale,
        max_half_width=_BOTTOM_MAX_HALF_WIDTH,
        grid_points=_GRID_POINTS,
        vmin=x_lo,
        vmax=x_hi,
    ):
        fig.add_trace(trace, row=3, col=2)

    fig.add_trace(
        go.Violin(
            y=shap_vals,
            orientation="v",
            side="both",
            points=False,
            box_visible=False,
            meanline_visible=True,
            spanmode="hard",
            line_color=SHAP_GRAY,
            fillcolor="rgba(119, 119, 119, 0.25)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Box(
            y=shap_vals,
            orientation="v",
            line_color=SHAP_GRAY,
            fillcolor="rgba(119, 119, 119, 0.25)",
            boxmean=True,
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    _layout_axes(fig, cfg, name)
    return fig


def _layout_axes(fig: go.Figure, cfg: ScatterRangesConfig, name: str) -> None:
    scatter_x, scatter_y = _axis_ref(2, 2, "x"), _axis_ref(2, 2, "y")

    # Top/scatter x-axes share the bottom violin's real-value axis but carry
    # no ticks of their own - Plotly renders a matched group's labels on the
    # bottom-most member regardless of per-axis showticklabels, so the title
    # and ticks are placed on row 3 to match where they actually land.
    fig.update_xaxes(matches=scatter_x, showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_xaxes(matches=scatter_x, showticklabels=False, showgrid=cfg.show_grid, row=2, col=2)
    fig.update_xaxes(matches=scatter_x, title_text=name, showgrid=False, row=3, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=2)

    # Left/right marginals share the scatter's SHAP-value y-axis but carry no
    # ticks of their own; their x-axis (density/box magnitude) is meaningless
    # and stays hidden.
    fig.update_yaxes(matches=scatter_y, showticklabels=False, showgrid=False, row=2, col=1)
    fig.update_yaxes(matches=scatter_y, showticklabels=False, showgrid=False, row=2, col=3)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=3)
    fig.update_yaxes(title_text=f"SHAP value for {name}", showgrid=cfg.show_grid, row=2, col=2)

    for row, col in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        title=cfg.title or f"SHAP dependence and value distributions - {name}",
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        showlegend=False,
    )
