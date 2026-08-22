"""Enumerations used across ``shaply`` to keep public options explicit and typed."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class PlotType(StrEnum):
    """Kind of SHAP figure that ``shaply`` can render."""

    BAR = "bar"
    BEESWARM = "beeswarm"
    BEESWARM_RANGES = "beeswarm_ranges"
    WATERFALL = "waterfall"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    FORCE = "force"
    DECISION = "decision"
    RESPONSE_CURVE = "response_curve"
    INTERACTION_HEATMAP = "interaction_heatmap"
    ERROR_ANALYSIS = "error_analysis"
    SHAP_SURFACE = "shap_surface"
    IMPORTANCE_BY_COHORT = "importance_by_cohort"
    FEATURE_CLUSTERING = "feature_clustering"
    EXPLANATION_ARCHETYPES = "explanation_archetypes"
    IMPORTANCE_CI = "importance_ci"
    MONOTONICITY = "monotonicity"


class FeatureOrdering(StrEnum):
    """Strategy used to order features along the categorical axis of a plot.

    Attributes
    ----------
    IMPORTANCE
        Order by mean absolute SHAP value (most important first).
    MAX_ABSOLUTE
        Order by the single largest absolute SHAP value across samples.
    ORIGINAL
        Keep the order of ``feature_names`` as provided.
    ALPHABETICAL
        Order features alphabetically by name.
    """

    IMPORTANCE = "importance"
    MAX_ABSOLUTE = "max_absolute"
    ORIGINAL = "original"
    ALPHABETICAL = "alphabetical"


class ColorScale(StrEnum):
    """Named color scales available for continuous encodings.

    ``RED_BLUE`` reproduces the canonical SHAP diverging scheme
    (blue for low feature values, red for high).
    """

    RED_BLUE = "red_blue"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    COOLWARM = "coolwarm"
    REDS = "reds"


class SortDirection(IntEnum):
    """Direction used when sorting numeric quantities."""

    ASCENDING = 1
    DESCENDING = -1
