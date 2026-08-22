"""Response-curve plot: smoothed SHAP effect of a feature with tipping points.

For a single feature this shows the smoothed mean SHAP value as a function of the
feature's value, a spread band, and the values where the mean effect crosses zero
(the "tipping points" where the feature switches from lowering to raising the
prediction). It is an actionable, PDP/ALE-flavored read of a dependence plot.

Note: SHAP effects are associational, not causal - read a tipping point as "the
value above which this feature is associated with a higher model output", not as
a proven physical cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import SHAP_GRAY, SHAP_RED
from shaply.config import ResponseCurveConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import resolve_feature_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Smoothed:
    """Binned summary of the feature-value to SHAP-value relationship."""

    centers: FloatArray
    means: FloatArray
    stds: FloatArray


def response_curve(
    values: ExplanationLike | npt.ArrayLike | object,
    feature: str | int,
    *,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ResponseCurveConfig | None = None,
) -> go.Figure:
    """Render the smoothed SHAP response curve of a single feature.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    feature
        Name or index of the feature to profile.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ResponseCurveConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The response-curve figure.

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
        msg = "response_curve requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    cfg = config or ResponseCurveConfig()
    return _build(explanation, cfg, feature)


def _bin_edges(x: FloatArray, n_bins: int, *, quantile: bool) -> FloatArray:
    if quantile:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(x, quantiles))
    else:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the max value
    return edges


def _smooth(x: FloatArray, y: FloatArray, cfg: ResponseCurveConfig) -> _Smoothed:
    """Aggregate ``y`` (SHAP) over bins of ``x`` (feature value)."""
    edges = _bin_edges(x, cfg.n_bins, quantile=cfg.quantile_bins)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    centers, means, stds = [], [], []
    for b in range(len(edges) - 1):
        members = y[idx == b]
        if members.size == 0:
            continue
        centers.append(float(x[idx == b].mean()))
        means.append(float(members.mean()))
        stds.append(float(members.std()))
    return _Smoothed(np.array(centers), np.array(means), np.array(stds))


def _zero_crossings(centers: FloatArray, means: FloatArray) -> list[float]:
    """Linearly interpolate the x-positions where the mean effect crosses zero."""
    crossings: list[float] = []
    for i in range(len(means) - 1):
        y0, y1 = means[i], means[i + 1]
        if y0 == 0.0:
            crossings.append(float(centers[i]))
        elif y0 * y1 < 0.0:
            t = y0 / (y0 - y1)
            crossings.append(float(centers[i] + t * (centers[i + 1] - centers[i])))
    return crossings


def _build(explanation: Explanation, cfg: ResponseCurveConfig, feature: str | int) -> go.Figure:
    assert explanation.data is not None  # guaranteed by caller
    idx = resolve_feature_index(explanation, feature)
    name = explanation.feature_names[idx]
    x = explanation.data[:, idx]
    y = explanation.values[:, idx]

    smoothed = _smooth(x, y, cfg)
    fig = go.Figure()

    if cfg.show_points:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"size": 5, "color": SHAP_GRAY, "opacity": cfg.point_opacity},
                name="samples",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if cfg.band and smoothed.centers.size:
        upper = smoothed.means + smoothed.stds
        lower = smoothed.means - smoothed.stds
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([smoothed.centers, smoothed.centers[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(255, 13, 87, 0.15)",
                line={"color": "rgba(0,0,0,0)"},
                name="±1 std",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=smoothed.centers,
            y=smoothed.means,
            mode="lines+markers",
            line={"color": SHAP_RED, "width": 2.5},
            name="mean SHAP",
            hovertemplate=f"{name} %{{x:.4g}}<br>mean SHAP %{{y:.4f}}<extra></extra>",
        )
    )

    apply_layout(
        fig,
        cfg,
        title=cfg.title or f"SHAP response curve - {name}",
        xaxis_title=name,
        yaxis_title=f"SHAP value for {name}",
    )
    fig.add_hline(y=0.0, line_color=SHAP_GRAY, line_width=1, opacity=0.6)

    if cfg.show_thresholds:
        for crossing in _zero_crossings(smoothed.centers, smoothed.means):
            fig.add_vline(
                x=crossing,
                line_color=SHAP_GRAY,
                line_width=1,
                line_dash="dash",
                annotation_text=f"tipping point ≈ {crossing:.3g}",
                annotation_position="top",
            )
    return fig
