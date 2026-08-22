"""Beeswarm summary plot (``shap.plots.beeswarm`` equivalent)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_GRAY, resolve_colorscale
from shaply.config import BeeswarmConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.enums import ColorScale
    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.intp]


def beeswarm(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: BeeswarmConfig | None = None,
) -> go.Figure:
    """Render a beeswarm summary plot as a Plotly figure.

    Each point is one sample's SHAP value for a feature, spread vertically by
    local density and colored by the (normalized) feature value.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.BeeswarmConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The beeswarm figure.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or BeeswarmConfig()
    return _build_beeswarm(explanation, cfg)


def _density_offsets(x: FloatArray, jitter: float, bins: int = 20) -> FloatArray:
    """Compute vertical offsets that spread overlapping points by local density."""
    if x.size == 0 or jitter == 0.0:
        return np.zeros_like(x)
    edges = np.linspace(x.min(), x.max() + 1e-9, bins + 1)
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    offsets = np.zeros_like(x)
    for b in range(bins):
        members = np.flatnonzero(bin_idx == b)
        count = members.size
        if count <= 1:
            continue
        # Symmetric spread in [-jitter, jitter] scaled by relative density.
        spread = np.linspace(-1.0, 1.0, count)
        density = min(1.0, count / (0.1 * x.size + 1.0))
        offsets[members] = spread * jitter * density
    return offsets


def _normalize_feature(column: FloatArray) -> FloatArray | None:
    """Scale a feature column to ``[0, 1]`` using its 5th-95th percentiles."""
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [5, 95])
    if high <= low:
        return None
    normalized: FloatArray = np.clip((column - low) / (high - low), 0.0, 1.0)
    return normalized


def beeswarm_scatter(
    explanation: Explanation,
    order: IntArray,
    color_scale: ColorScale,
    *,
    point_size: float,
    opacity: float,
    jitter: float,
    show_colorbar: bool = True,
) -> go.Scatter:
    """Build the beeswarm scatter trace for features given in display order.

    ``order`` lists feature indices bottom-to-top; each feature's samples are
    plotted at ``y = row (+ density offset)`` with ``x`` the SHAP value and the
    color encoding the (normalized) feature value.

    Parameters
    ----------
    explanation
        The explanation to render.
    order
        Feature indices in display order (bottom row first).
    color_scale
        Color scale encoding feature values.
    point_size, opacity, jitter
        Marker size, marker opacity and vertical density spread.
    show_colorbar
        Whether to attach the shared Low/High colorbar to the trace.

    Returns
    -------
    plotly.graph_objects.Scatter
        The assembled scatter trace.
    """
    has_color = explanation.data is not None
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    colors: list[FloatArray] = []
    customdata: list[FloatArray] = []

    for row, feature_idx in enumerate(order):
        shap_col = explanation.values[:, feature_idx]
        offsets = _density_offsets(shap_col, jitter)
        xs.append(shap_col)
        ys.append(np.full_like(shap_col, row) + offsets)
        if has_color and explanation.data is not None:
            raw = explanation.data[:, feature_idx]
            normalized = _normalize_feature(raw)
            colors.append(normalized if normalized is not None else np.full_like(raw, 0.5))
            customdata.append(raw)

    marker: dict[str, object] = {"size": point_size, "opacity": opacity}
    if has_color:
        marker.update(
            {
                "color": np.concatenate(colors),
                "colorscale": resolve_colorscale(color_scale),
                "cmin": 0.0,
                "cmax": 1.0,
            }
        )
        if show_colorbar:
            marker["colorbar"] = {
                "title": {"text": "Feature value", "side": "right"},
                "tickmode": "array",
                "tickvals": [0.0, 1.0],
                "ticktext": ["Low", "High"],
            }
        hovertemplate = "SHAP %{x:.4f}<br>value %{customdata:.4g}<extra></extra>"
        custom = np.concatenate(customdata)
    else:
        marker["color"] = SHAP_GRAY
        hovertemplate = "SHAP %{x:.4f}<extra></extra>"
        custom = None

    return go.Scatter(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        mode="markers",
        marker=marker,
        customdata=custom,
        hovertemplate=hovertemplate,
    )


def _build_beeswarm(explanation: Explanation, cfg: BeeswarmConfig) -> go.Figure:
    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    # Display order bottom-to-top: least important at the bottom.
    order = layout.order[::-1]
    labels = layout.labels[::-1]

    fig = go.Figure(
        beeswarm_scatter(
            explanation,
            order,
            cfg.color_scale,
            point_size=cfg.point_size,
            opacity=cfg.opacity,
            jitter=cfg.jitter,
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=list(labels),
        showgrid=False,
    )
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "SHAP summary (beeswarm)",
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title=None,
    )
    fig.add_vline(x=0.0, line_color=SHAP_GRAY, line_width=1, opacity=0.5)
    return fig
