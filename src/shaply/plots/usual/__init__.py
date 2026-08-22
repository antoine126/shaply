"""The usual SHAP figures: bar, beeswarm, waterfall, scatter, heatmap, force, decision."""

from __future__ import annotations

from shaply.plots.usual.bar import bar
from shaply.plots.usual.beeswarm import beeswarm
from shaply.plots.usual.decision import decision
from shaply.plots.usual.force import force
from shaply.plots.usual.heatmap import heatmap
from shaply.plots.usual.scatter import scatter
from shaply.plots.usual.waterfall import waterfall

__all__ = [
    "bar",
    "beeswarm",
    "decision",
    "force",
    "heatmap",
    "scatter",
    "waterfall",
]
