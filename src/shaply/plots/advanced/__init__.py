"""shaply-only advanced tools: beyond the usual SHAP figures.

Cross SHAP values with the real data to surface operating ranges, tipping
points, coupled effects, redundancy, archetypes and robustness - read as
associational, not causal.
"""

from __future__ import annotations

from shaply.plots.advanced.beeswarm_ranges import beeswarm_ranges
from shaply.plots.advanced.error_analysis import error_analysis
from shaply.plots.advanced.explanation_archetypes import explanation_archetypes
from shaply.plots.advanced.feature_clustering import feature_clustering
from shaply.plots.advanced.importance_by_cohort import importance_by_cohort
from shaply.plots.advanced.importance_ci import importance_ci
from shaply.plots.advanced.interaction_heatmap import interaction_heatmap
from shaply.plots.advanced.monotonicity import monotonicity_check
from shaply.plots.advanced.response_curve import response_curve
from shaply.plots.advanced.shap_surface import shap_surface

__all__ = [
    "beeswarm_ranges",
    "error_analysis",
    "explanation_archetypes",
    "feature_clustering",
    "importance_by_cohort",
    "importance_ci",
    "interaction_heatmap",
    "monotonicity_check",
    "response_curve",
    "shap_surface",
]
