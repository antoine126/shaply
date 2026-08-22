"""Plotly figure builders for the usual SHAP explanations and advanced tools.

Organized in three subpackages:

- ``shaply.plots._common`` - private, dependency-free helpers shared by all plots.
- ``shaply.plots.usual`` - the familiar ``shap.plots`` equivalents.
- ``shaply.plots.advanced`` - shaply-only tools that go beyond the usual figures.
"""

from __future__ import annotations

from shaply.plots.advanced import (
    beeswarm_ranges,
    error_analysis,
    explanation_archetypes,
    feature_clustering,
    importance_by_cohort,
    importance_ci,
    interaction_heatmap,
    monotonicity_check,
    response_curve,
    shap_surface,
)
from shaply.plots.usual import (
    bar,
    beeswarm,
    decision,
    force,
    heatmap,
    scatter,
    waterfall,
)

__all__ = [
    "bar",
    "beeswarm",
    "beeswarm_ranges",
    "decision",
    "error_analysis",
    "explanation_archetypes",
    "feature_clustering",
    "force",
    "heatmap",
    "importance_by_cohort",
    "importance_ci",
    "interaction_heatmap",
    "monotonicity_check",
    "response_curve",
    "scatter",
    "shap_surface",
    "waterfall",
]
