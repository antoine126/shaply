"""Shared Plotly layout application for all figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from shaply.config import BasePlotConfig


def apply_layout(
    fig: go.Figure,
    cfg: BasePlotConfig,
    *,
    title: str | None,
    xaxis_title: str | None,
    yaxis_title: str | None,
) -> None:
    """Apply shared configuration (title, size, template, grid) to ``fig``.

    Parameters
    ----------
    fig
        The figure to mutate in place.
    cfg
        The plot configuration carrying shared layout options.
    title, xaxis_title, yaxis_title
        Resolved axis and figure titles for this specific plot.
    """
    fig.update_layout(
        title=title,
        template=cfg.template,
        width=cfg.width,
        height=cfg.height,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    fig.update_xaxes(showgrid=cfg.show_grid, zeroline=False)
    fig.update_yaxes(showgrid=cfg.show_grid, zeroline=False)
