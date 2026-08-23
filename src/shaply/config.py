"""Public, validated plot configuration models (pydantic v2).

These are the external data models users can pass to tune figures. They are
deliberately kept separate from the internal :mod:`shaply.explanation`
dataclasses: pydantic validates and normalizes user input, dataclasses carry
already-trusted internal state.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shaply.enums import ColorScale, FeatureOrdering

_DEFAULT_TEMPLATE = "plotly_white"


class BasePlotConfig(BaseModel):
    """Options shared by every ``shaply`` figure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str | None = Field(default=None, description="Figure title.")
    width: int | None = Field(default=None, gt=0, description="Figure width in pixels.")
    height: int | None = Field(default=None, gt=0, description="Figure height in pixels.")
    template: str = Field(default=_DEFAULT_TEMPLATE, description="Plotly layout template.")
    show_grid: bool = Field(default=True, description="Whether to draw axis gridlines.")


class BarConfig(BasePlotConfig):
    """Configuration for the global feature-importance bar plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features along the axis.",
    )
    show_values: bool = Field(default=True, description="Annotate each bar with its numeric value.")


class BeeswarmConfig(BasePlotConfig):
    """Configuration for the beeswarm (summary) plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features along the axis.",
    )
    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Color scale encoding feature values.",
    )
    point_size: float = Field(default=5.0, gt=0, description="Marker size in pixels.")
    jitter: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Vertical spread of overlapping points."
    )
    opacity: float = Field(default=0.8, gt=0.0, le=1.0, description="Marker opacity.")


class BeeswarmRangesConfig(BeeswarmConfig):
    """Configuration for the combined beeswarm + feature-value-ranges figure.

    Extends :class:`BeeswarmConfig` with the options controlling the right-hand
    panel that shows the real distribution (violin + box) of each feature's
    values.
    """

    impact_panel_ratio: float = Field(
        default=0.62,
        gt=0.2,
        lt=0.9,
        description="Fraction of width given to the left (SHAP impact) panel.",
    )
    show_value_labels: bool = Field(
        default=True,
        description="Annotate each feature row with its real min and max value.",
    )


class WaterfallConfig(BasePlotConfig):
    """Configuration for the single-prediction waterfall plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    show_values: bool = Field(default=True, description="Annotate each step with its contribution.")


class ScatterConfig(BasePlotConfig):
    """Configuration for the dependence / scatter plot."""

    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Color scale used when coloring by an interaction feature.",
    )
    point_size: float = Field(default=6.0, gt=0, description="Marker size in pixels.")
    opacity: float = Field(default=0.8, gt=0.0, le=1.0, description="Marker opacity.")


class ScatterRangesConfig(BasePlotConfig):
    """Configuration for the dependence scatter framed by marginal distributions."""

    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Color scale for the points and the two colored marginals.",
    )
    point_size: float = Field(default=6.0, gt=0, description="Marker size in pixels.")
    opacity: float = Field(default=0.7, gt=0.0, le=1.0, description="Marker opacity.")
    marginal_ratio: float = Field(
        default=0.16,
        gt=0.05,
        lt=0.35,
        description="Fraction of width/height given to each marginal panel.",
    )


class HeatmapConfig(BasePlotConfig):
    """Configuration for the instances-by-features heatmap."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features along the axis.",
    )
    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Color scale encoding SHAP values.",
    )


class ResponseCurveConfig(BasePlotConfig):
    """Configuration for the response-curve plot (smoothed effect + thresholds)."""

    n_bins: int = Field(default=20, gt=2, description="Number of bins used to smooth.")
    quantile_bins: bool = Field(
        default=True,
        description="Use equal-count quantile bins (stable) instead of equal-width.",
    )
    band: bool = Field(default=True, description="Draw a +/-1 std spread band.")
    show_points: bool = Field(default=True, description="Overlay the raw scatter points.")
    point_opacity: float = Field(default=0.25, gt=0.0, le=1.0, description="Raw point opacity.")
    show_thresholds: bool = Field(
        default=True, description="Mark the values where the mean effect crosses zero."
    )


class InteractionHeatmapConfig(BasePlotConfig):
    """Configuration for the pairwise SHAP-interaction heatmap."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    color_scale: ColorScale = Field(
        default=ColorScale.REDS,
        description="Sequential color scale encoding interaction magnitude.",
    )
    show_diagonal: bool = Field(
        default=False,
        description="Keep the diagonal (main effects); hidden by default to reveal interactions.",
    )
    show_values: bool = Field(default=False, description="Annotate each cell with its value.")


class ErrorAnalysisConfig(BasePlotConfig):
    """Configuration for the error-analysis plot (SHAP drivers of errors)."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    show_values: bool = Field(default=True, description="Annotate each bar with its value.")


class ShapSurfaceConfig(BasePlotConfig):
    """Configuration for the 2D SHAP interaction surface."""

    n_bins: int = Field(default=20, gt=2, description="Grid resolution per axis.")
    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Diverging color scale encoding the mean SHAP value.",
    )
    contour: bool = Field(default=False, description="Render smooth contours instead of a heatmap.")


class ImportanceByCohortConfig(BasePlotConfig):
    """Configuration for the cohort-wise importance plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features (by overall importance across cohorts).",
    )
    n_cohorts: int = Field(
        default=3, gt=1, description="Number of quantile cohorts when splitting by a feature."
    )


class FeatureClusteringConfig(BasePlotConfig):
    """Configuration for the SHAP-similarity feature-clustering heatmap."""

    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Diverging color scale encoding SHAP correlation.",
    )
    show_values: bool = Field(default=False, description="Annotate each cell with its value.")


class ExplanationArchetypesConfig(BasePlotConfig):
    """Configuration for the SHAP-profile archetypes plot."""

    n_clusters: int = Field(default=4, gt=1, description="Number of archetypes to extract.")
    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features along the axis.",
    )
    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Diverging color scale encoding each archetype's mean SHAP.",
    )
    random_state: int = Field(default=0, description="Seed for the k-means clustering.")


class ImportanceCIConfig(BasePlotConfig):
    """Configuration for the bootstrap importance plot with confidence intervals."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features along the axis.",
    )
    n_boot: int = Field(default=1000, gt=10, description="Number of bootstrap resamples.")
    ci: float = Field(default=0.95, gt=0.0, lt=1.0, description="Confidence level.")
    random_state: int = Field(default=0, description="Seed for the bootstrap resampling.")


class MonotonicityConfig(BasePlotConfig):
    """Configuration for the monotonicity-check plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")


class ForceConfig(BasePlotConfig):
    """Configuration for the single-prediction force plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    show_values: bool = Field(
        default=True, description="Annotate each segment with its contribution."
    )


class DecisionConfig(BasePlotConfig):
    """Configuration for the decision plot."""

    max_display: int = Field(default=10, gt=0, description="Max features to display.")
    ordering: FeatureOrdering = Field(
        default=FeatureOrdering.IMPORTANCE,
        description="How to order features along the axis.",
    )
    color_scale: ColorScale = Field(
        default=ColorScale.RED_BLUE,
        description="Color scale encoding each instance's predicted output.",
    )
    line_width: float = Field(default=1.5, gt=0, description="Line width in pixels.")
    opacity: float = Field(default=0.8, gt=0.0, le=1.0, description="Line opacity.")
