"""shaply - usual SHAP explainability figures rendered as Plotly charts.

The public API mirrors the familiar ``shap.plots`` entry points but returns
:class:`plotly.graph_objects.Figure` objects instead of matplotlib axes::

    import shaply

    fig = shaply.beeswarm(shap_values)   # a shap.Explanation, ndarray or DataFrame
    fig.show()

Every plotting function accepts a ``shap.Explanation``-like object, a numpy
array of SHAP values, or a :class:`pandas.DataFrame`, and an optional typed
config object from :mod:`shaply.config`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from shaply.config import (
    BarConfig,
    BeeswarmConfig,
    BeeswarmRangesConfig,
    DecisionConfig,
    ErrorAnalysisConfig,
    ExplanationArchetypesConfig,
    FeatureClusteringConfig,
    ForceConfig,
    HeatmapConfig,
    ImportanceByCohortConfig,
    ImportanceCIConfig,
    InteractionHeatmapConfig,
    MonotonicityConfig,
    ResponseCurveConfig,
    ScatterConfig,
    ScatterRangesConfig,
    ShapSurfaceConfig,
    WaterfallConfig,
)
from shaply.enums import ColorScale, FeatureOrdering, PlotType
from shaply.explanation import Explanation, to_explanation
from shaply.interaction import InteractionValues, to_interaction_values
from shaply.plots import (
    bar,
    beeswarm,
    beeswarm_ranges,
    decision,
    error_analysis,
    explanation_archetypes,
    feature_clustering,
    force,
    heatmap,
    importance_by_cohort,
    importance_ci,
    interaction_heatmap,
    monotonicity_check,
    response_curve,
    scatter,
    scatter_ranges,
    shap_surface,
    waterfall,
)

try:
    __version__ = version("shaply")
except PackageNotFoundError:  # pragma: no cover - only during local dev without install
    __version__ = "0.0.0"

__all__ = [
    "BarConfig",
    "BeeswarmConfig",
    "BeeswarmRangesConfig",
    "ColorScale",
    "DecisionConfig",
    "ErrorAnalysisConfig",
    "Explanation",
    "ExplanationArchetypesConfig",
    "FeatureClusteringConfig",
    "FeatureOrdering",
    "ForceConfig",
    "HeatmapConfig",
    "ImportanceByCohortConfig",
    "ImportanceCIConfig",
    "InteractionHeatmapConfig",
    "InteractionValues",
    "MonotonicityConfig",
    "PlotType",
    "ResponseCurveConfig",
    "ScatterConfig",
    "ScatterRangesConfig",
    "ShapSurfaceConfig",
    "WaterfallConfig",
    "__version__",
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
    "scatter_ranges",
    "shap_surface",
    "to_explanation",
    "to_interaction_values",
    "waterfall",
]
