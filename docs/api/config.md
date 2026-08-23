# Configuration

Every plotting function accepts an optional typed `config` object to override its defaults
(colors, ordering, layout, ...). All config classes are Pydantic models and validate on
construction.

## Usual plots

::: shaply.BarConfig

::: shaply.BeeswarmConfig

::: shaply.WaterfallConfig

::: shaply.ScatterConfig

::: shaply.HeatmapConfig

::: shaply.DecisionConfig

::: shaply.ForceConfig

## Advanced plots

::: shaply.BeeswarmRangesConfig

::: shaply.ErrorAnalysisConfig

::: shaply.ExplanationArchetypesConfig

::: shaply.FeatureClusteringConfig

::: shaply.ImportanceByCohortConfig

::: shaply.ImportanceCIConfig

::: shaply.InteractionHeatmapConfig

::: shaply.MonotonicityConfig

::: shaply.ResponseCurveConfig

::: shaply.ScatterRangesConfig

::: shaply.ShapSurfaceConfig
