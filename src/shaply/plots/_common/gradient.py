"""Gradient-colored KDE silhouette and box marginal shapes.

Plotly's ``Violin``/``Box`` traces only support a single flat fill color, so
when a figure wants a density silhouette or a box colored with the same
low-to-high scale used elsewhere (e.g. the beeswarm dots), it has to be
hand-drawn: a Gaussian KDE evaluated on a grid, sliced into many thin colored
bands. This module holds that machinery so it is shared rather than
duplicated across plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_GRAY, sample_scale

if TYPE_CHECKING:
    import numpy.typing as npt

    from shaply.enums import ColorScale

    FloatArray = npt.NDArray[np.float64]

_BOX_LINE_COLOR = "#333333"


def kde_density(values: FloatArray, grid: FloatArray) -> FloatArray:
    """Gaussian KDE of ``values`` evaluated on ``grid`` (same units as values)."""
    n = values.size
    std = float(values.std())
    if n < 2 or std == 0.0:
        return np.zeros_like(grid)
    bandwidth = max(1.06 * std * n ** (-1 / 5), 1e-9)  # Silverman's rule of thumb
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    density: FloatArray = np.exp(-0.5 * diffs**2).sum(axis=1)
    density /= n * bandwidth * np.sqrt(2 * np.pi)
    return density


def _color_positions(grid: FloatArray, vmin: float, vmax: float) -> list[float]:
    if vmax <= vmin:
        return [0.5] * grid.size
    return [float(v) for v in (grid - vmin) / (vmax - vmin)]


def gradient_silhouette_traces(
    values: FloatArray,
    baseline: float,
    color_scale: ColorScale,
    *,
    orientation: str = "h",
    max_half_width: float = 0.42,
    grid_points: int = 31,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[go.Scatter]:
    """Gradient-filled density silhouette: many colored bands + a thin outline.

    Parameters
    ----------
    values
        The raw samples the KDE is computed from.
    baseline
        The row (``orientation="h"``) or column (``orientation="v"``) position
        the silhouette is centered on.
    color_scale
        Color scale sampled low-to-high along the value axis to color each
        band, matching the convention used elsewhere (e.g. the beeswarm dots).
    orientation
        ``"h"``: ``values`` run along x, density spreads around ``baseline``
        on y. ``"v"``: ``values`` run along y, density spreads around
        ``baseline`` on x.
    max_half_width
        Half-extent of the silhouette at its widest point.
    grid_points
        Number of KDE evaluation points; ``grid_points - 1`` color bands.
    vmin, vmax
        Value-axis bounds for the KDE grid and the color mapping; default to
        the data's own min/max.

    Returns
    -------
    list of go.Scatter
        The colored band traces plus a final thin outline trace.
    """
    lo = float(values.min()) if vmin is None else vmin
    hi = float(values.max()) if vmax is None else vmax
    if hi <= lo:
        return []
    grid: FloatArray = np.linspace(lo, hi, grid_points)
    density = kde_density(values, grid)
    peak = float(density.max())
    half_width = density / peak * max_half_width if peak > 0 else np.zeros_like(density)
    spread_top = baseline + half_width
    spread_bottom = baseline - half_width

    positions = _color_positions(grid, lo, hi)
    midpoints = [(positions[i] + positions[i + 1]) / 2 for i in range(len(positions) - 1)]
    band_colors = sample_scale(color_scale, midpoints)

    def _band(i: int) -> go.Scatter:
        value_edges = [grid[i], grid[i + 1], grid[i + 1], grid[i]]
        spread_edges = [spread_bottom[i], spread_bottom[i + 1], spread_top[i + 1], spread_top[i]]
        x, y = (value_edges, spread_edges) if orientation == "h" else (spread_edges, value_edges)
        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=band_colors[i],
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )

    traces = [_band(i) for i in range(grid_points - 1)]

    outline_value = np.concatenate([grid, grid[::-1]])
    outline_spread = np.concatenate([spread_top, spread_bottom[::-1]])
    ox, oy = (
        (outline_value, outline_spread) if orientation == "h" else (outline_spread, outline_value)
    )
    traces.append(
        go.Scatter(
            x=ox,
            y=oy,
            mode="lines",
            line={"color": SHAP_GRAY, "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return traces


def gradient_box_trace(
    values: FloatArray,
    baseline: float,
    color_scale: ColorScale | None,
    *,
    orientation: str = "h",
    half_height: float = 0.11,
    vmin: float | None = None,
    vmax: float | None = None,
    n_bands: int = 20,
) -> list[go.Scatter]:
    """Quartile box + whiskers + median/mean ticks, optionally gradient-filled.

    Parameters
    ----------
    values
        The raw samples the quartiles/whiskers are computed from.
    baseline
        The row (``orientation="h"``) or column (``orientation="v"``) position
        the box is centered on.
    color_scale
        Pass ``None`` for a flat, uncolored outline-only box. Pass a scale to
        fill the Q1-Q3 body with ``n_bands`` color-interpolated bands, using
        the same value-axis bounds (``vmin``/``vmax``) as a companion
        silhouette so the two line up.
    orientation
        Same convention as :func:`gradient_silhouette_traces`.
    half_height
        Half-extent of the box along the spread axis.
    vmin, vmax
        Value-axis bounds for the color mapping; default to the data's own
        min/max. Ignored when ``color_scale`` is ``None``.
    n_bands
        Number of color bands filling the box body.

    Returns
    -------
    list of go.Scatter
        The colored band traces (if any) plus a final outline trace.
    """
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    within = values[(values >= lower_fence) & (values <= upper_fence)]
    whisker_lo = float(within.min()) if within.size else float(values.min())
    whisker_hi = float(within.max()) if within.size else float(values.max())
    mean = float(values.mean())
    half = half_height

    traces: list[go.Scatter] = []
    if color_scale is not None and q3 > q1:
        lo = float(values.min()) if vmin is None else vmin
        hi = float(values.max()) if vmax is None else vmax
        edges = np.linspace(q1, q3, n_bands + 1)
        positions = _color_positions(edges, lo, hi)
        midpoints = [(positions[i] + positions[i + 1]) / 2 for i in range(n_bands)]
        band_colors = sample_scale(color_scale, midpoints)
        for i in range(n_bands):
            value_edges = [edges[i], edges[i + 1], edges[i + 1], edges[i]]
            spread_edges = [baseline - half, baseline - half, baseline + half, baseline + half]
            x, y = (
                (value_edges, spread_edges)
                if orientation == "h"
                else (spread_edges, value_edges)
            )
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=band_colors[i],
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Outline: whisker, box border, median tick, gap, mean tick - one polyline.
    v = [whisker_lo, q1, None, q1, q3, q3, q1, q1, None, q3, whisker_hi, None]
    s = [baseline, baseline, None, baseline - half, baseline - half, baseline + half]
    s += [baseline + half, baseline - half, None, baseline, baseline, None]
    v += [median, median, None, mean, mean]
    s += [baseline - half, baseline + half, None, baseline - half * 0.7, baseline + half * 0.7]
    x, y = (v, s) if orientation == "h" else (s, v)
    traces.append(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": _BOX_LINE_COLOR, "width": 1.3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return traces
