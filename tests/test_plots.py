from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import shaply
from shaply.config import BarConfig, BeeswarmConfig, DecisionConfig, HeatmapConfig
from shaply.enums import FeatureOrdering
from tests.conftest import FakeExplanation


def test_bar_returns_figure(fake_explanation: FakeExplanation) -> None:
    fig = shaply.bar(fake_explanation)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"


def test_bar_respects_max_display(shap_values: np.ndarray) -> None:
    fig = shaply.bar(shap_values, config=BarConfig(max_display=3))
    # 2 top features + 1 grouped row.
    assert len(fig.data[0].y) == 3
    assert any("other features" in str(label) for label in fig.data[0].y)


def test_bar_single_instance_is_signed() -> None:
    fig = shaply.bar(np.array([1.0, -2.0, 0.5]))
    assert fig.data[0].type == "bar"


def test_beeswarm_has_colorbar_when_data_present(
    fake_explanation: FakeExplanation,
) -> None:
    fig = shaply.beeswarm(fake_explanation)
    assert fig.data[0].marker.colorscale is not None


def test_beeswarm_without_data_is_gray(shap_values: np.ndarray) -> None:
    fig = shaply.beeswarm(shap_values, config=BeeswarmConfig(max_display=4))
    assert fig.data[0].marker.colorscale is None


def test_waterfall_single_prediction(fake_explanation: FakeExplanation) -> None:
    fig = shaply.waterfall(fake_explanation, sample_index=2)
    assert fig.data[0].type == "waterfall"


def test_scatter_requires_data(shap_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="requires feature values"):
        shaply.scatter(shap_values, feature=0)


def test_scatter_by_name_and_color(fake_explanation: FakeExplanation) -> None:
    fig = shaply.scatter(fake_explanation, feature="income", color_feature="age")
    assert fig.data[0].type == "scatter"
    assert fig.data[0].marker.colorscale is not None


def test_scatter_unknown_feature(fake_explanation: FakeExplanation) -> None:
    with pytest.raises(KeyError):
        shaply.scatter(fake_explanation, feature="does_not_exist")


def test_heatmap_shape(fake_explanation: FakeExplanation) -> None:
    fig = shaply.heatmap(fake_explanation, config=HeatmapConfig(max_display=4))
    assert fig.data[0].type == "heatmap"
    assert np.asarray(fig.data[0].z).shape == (4, 50)


def test_force_single_prediction(fake_explanation: FakeExplanation) -> None:
    fig = shaply.force(fake_explanation, sample_index=1)
    assert fig.data[0].type == "bar"
    # Segment widths are non-negative and the strip reaches the prediction.
    assert all(w >= 0 for w in fig.data[0].x)


def test_force_conserves_prediction() -> None:
    # base + sum(shap) must equal the right edge of the last red segment / strip.
    values = np.array([2.0, -1.0, 0.5])
    fig = shaply.force(values, base_values=1.0)
    base = np.asarray(fig.data[0].base, dtype=float)
    widths = np.asarray(fig.data[0].x, dtype=float)
    right_edges = base + widths
    left_edges = base
    prediction = 1.0 + values.sum()
    assert np.isclose(right_edges.max(), 1.0 + values[values > 0].sum())
    assert np.isclose(left_edges.min(), 1.0 + values[values < 0].sum())
    # The red/blue boundary sits at the prediction.
    assert min(abs(right_edges - prediction).min(), abs(left_edges - prediction).min()) < 1e-9


def test_decision_returns_lines(fake_explanation: FakeExplanation) -> None:
    fig = shaply.decision(fake_explanation, config=DecisionConfig(max_display=4))
    assert len(fig.data) == 50  # one line per instance
    assert fig.data[0].type == "scatter"
    # Each path has base level + displayed features.
    assert len(fig.data[0].x) == 5


def test_decision_path_ends_at_prediction() -> None:
    values = np.array([[1.0, -2.0, 0.5]])
    fig = shaply.decision(values, base_values=0.5)
    path = np.asarray(fig.data[0].x, dtype=float)
    assert np.isclose(path[0], 0.5)  # starts at base
    assert np.isclose(path[-1], 0.5 + values.sum())  # ends at f(x)


def test_beeswarm_ranges_has_two_panels(fake_explanation: FakeExplanation) -> None:
    fig = shaply.beeswarm_ranges(fake_explanation)
    kinds = [d.type for d in fig.data]
    assert "violin" not in kinds  # replaced by a hand-drawn gradient silhouette
    assert kinds.count("scatter") == len(kinds)  # left impact panel + right silhouettes
    assert len(kinds) > 4 * 10  # several color bands per feature row, not one shape each
    # The silhouette is colored with the beeswarm's low->high scale, not a flat gray.
    fill_colors = {d.fillcolor for d in fig.data if d.fill == "toself" and d.fillcolor}
    assert len(fill_colors) > 4  # more than one color per feature => an actual gradient
    # Real min/max annotations are present for each feature (2 per row).
    range_labels = [a.text for a in fig.layout.annotations if a.xref == "x2"]
    assert len(range_labels) == 8


def test_beeswarm_ranges_requires_data(shap_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="requires feature values"):
        shaply.beeswarm_ranges(shap_values)


def test_ordering_alphabetical(fake_explanation: FakeExplanation) -> None:
    fig = shaply.bar(
        fake_explanation,
        config=BarConfig(ordering=FeatureOrdering.ALPHABETICAL, max_display=10),
    )
    # Top-to-bottom is reversed for display; reverse back to check sort.
    labels = list(fig.data[0].y)[::-1]
    assert labels == sorted(labels)
