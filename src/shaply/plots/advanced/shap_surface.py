"""2D SHAP interaction surface over the plane of two features.

Bins the ``(feature_x, feature_y)`` plane and shows the mean SHAP value of one
feature in each cell. Bright/dark regions expose *operating regions* where the
feature helps or hurts the prediction, and how a second feature modulates it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.colors import resolve_colorscale
from shaply.config import ShapSurfaceConfig
from shaply.explanation import to_explanation
from shaply.plots._common.ordering import resolve_feature_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    FloatArray = npt.NDArray[np.float64]


def shap_surface(
    values: ExplanationLike | npt.ArrayLike | object,
    feature_x: str | int,
    feature_y: str | int,
    *,
    shap_of: str | int | None = None,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ShapSurfaceConfig | None = None,
) -> go.Figure:
    """Render the mean SHAP surface over the plane of two features.

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    feature_x, feature_y
        Features spanning the horizontal and vertical axes.
    shap_of
        Which feature's SHAP value to average in each cell; defaults to
        ``feature_x``.
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ShapSurfaceConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The surface figure.

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
        msg = "shap_surface requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    cfg = config or ShapSurfaceConfig()
    return _build(explanation, cfg, feature_x, feature_y, shap_of)


def _grid_means(
    x: FloatArray, y: FloatArray, z: FloatArray, n_bins: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(x_centers, y_centers, mean_z)`` binned over the (x, y) plane."""
    x_edges = np.linspace(x.min(), np.nextafter(x.max(), np.inf), n_bins + 1)
    y_edges = np.linspace(y.min(), np.nextafter(y.max(), np.inf), n_bins + 1)
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

    grid = np.full((n_bins, n_bins), np.nan)
    for gy in range(n_bins):
        for gx in range(n_bins):
            members = z[(xi == gx) & (yi == gy)]
            if members.size:
                grid[gy, gx] = members.mean()

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return x_centers, y_centers, grid


def _build(
    explanation: Explanation,
    cfg: ShapSurfaceConfig,
    feature_x: str | int,
    feature_y: str | int,
    shap_of: str | int | None,
) -> go.Figure:
    assert explanation.data is not None  # guaranteed by caller
    ix = resolve_feature_index(explanation, feature_x)
    iy = resolve_feature_index(explanation, feature_y)
    iz = ix if shap_of is None else resolve_feature_index(explanation, shap_of)

    x = explanation.data[:, ix]
    y = explanation.data[:, iy]
    z = explanation.values[:, iz]
    x_centers, y_centers, grid = _grid_means(x, y, z, cfg.n_bins)

    bound = float(np.nanmax(np.abs(grid))) or 1.0
    colorscale = resolve_colorscale(cfg.color_scale)
    shared = {
        "x": x_centers,
        "y": y_centers,
        "z": grid,
        "colorscale": colorscale,
        "zmid": 0.0,
        "zmin": -bound,
        "zmax": bound,
        "colorbar": {"title": {"text": f"mean SHAP<br>of {explanation.feature_names[iz]}"}},
    }
    trace = (
        go.Contour(**shared, connectgaps=True, line={"width": 0})
        if cfg.contour
        else go.Heatmap(**shared)
    )
    fig = go.Figure(trace)
    fig.update_layout(
        title=cfg.title
        or f"SHAP surface - {explanation.feature_names[ix]} × {explanation.feature_names[iy]}",  # noqa: RUF001
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        xaxis_title=explanation.feature_names[ix],
        yaxis_title=explanation.feature_names[iy],
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig
