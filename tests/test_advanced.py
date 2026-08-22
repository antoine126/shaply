from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import shaply
from shaply.config import InteractionHeatmapConfig, ResponseCurveConfig
from shaply.interaction import InteractionValues, to_interaction_values
from tests.conftest import FakeExplanation


# --------------------------------------------------------------------------- #
# response_curve
# --------------------------------------------------------------------------- #
def test_response_curve_requires_data(shap_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="requires feature values"):
        shaply.response_curve(shap_values, feature=0)


def test_response_curve_detects_tipping_point() -> None:
    # A feature whose SHAP effect flips sign at x = 0 -> one zero crossing.
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 400)
    sv = np.column_stack([x + rng.normal(0, 0.05, x.size), rng.normal(0, 0.1, x.size)])
    data = np.column_stack([x, rng.normal(0, 1, x.size)])
    fig = shaply.response_curve(
        sv,
        feature=0,
        data=data,
        feature_names=["signal", "noise"],
        config=ResponseCurveConfig(show_points=False),
    )
    assert isinstance(fig, go.Figure)
    crossing_notes = [
        a.text for a in fig.layout.annotations if a.text and "tipping point" in a.text
    ]
    assert len(crossing_notes) >= 1


def test_response_curve_mean_line_present(fake_explanation: FakeExplanation) -> None:
    fig = shaply.response_curve(fake_explanation, feature="age")
    names = [tr.name for tr in fig.data]
    assert "mean SHAP" in names


# --------------------------------------------------------------------------- #
# interaction_heatmap
# --------------------------------------------------------------------------- #
def _interaction_tensor(n: int = 30, f: int = 4) -> np.ndarray:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(n, f, f))
    return (raw + raw.transpose(0, 2, 1)) / 2  # symmetric interactions


def test_to_interaction_values_validates_shape() -> None:
    with pytest.raises(ValueError, match="n_features, n_features"):
        InteractionValues(values=np.zeros((5, 3, 2)), feature_names=("a", "b", "c"))


def test_to_interaction_values_defaults_names() -> None:
    iv = to_interaction_values(_interaction_tensor(f=3))
    assert iv.feature_names == ("Feature 0", "Feature 1", "Feature 2")
    assert iv.mean_abs_matrix().shape == (3, 3)


def test_interaction_heatmap_hides_diagonal_by_default() -> None:
    fig = shaply.interaction_heatmap(_interaction_tensor(), feature_names=list("abcd"))
    z = np.asarray(fig.data[0].z, dtype=float)
    assert np.all(np.isnan(np.diag(z)))  # diagonal masked


def test_interaction_heatmap_max_display() -> None:
    fig = shaply.interaction_heatmap(
        _interaction_tensor(f=6), config=InteractionHeatmapConfig(max_display=3)
    )
    assert np.asarray(fig.data[0].z).shape == (3, 3)


# --------------------------------------------------------------------------- #
# error_analysis
# --------------------------------------------------------------------------- #
def test_error_analysis_from_mask(fake_explanation: FakeExplanation) -> None:
    mask = np.zeros(50, dtype=bool)
    mask[:10] = True
    fig = shaply.error_analysis(fake_explanation, errors=mask)
    assert [tr.name for tr in fig.data] == ["correct", "error"]


def test_error_analysis_from_labels(fake_explanation: FakeExplanation) -> None:
    y_true = np.zeros(50, dtype=int)
    y_pred = np.zeros(50, dtype=int)
    y_pred[:5] = 1  # five mistakes
    fig = shaply.error_analysis(fake_explanation, y_true=y_true, y_pred=y_pred)
    assert fig.data[0].type == "bar"


def test_error_analysis_requires_cohort(fake_explanation: FakeExplanation) -> None:
    with pytest.raises(ValueError, match="either errors"):
        shaply.error_analysis(fake_explanation)


def test_error_analysis_rejects_single_cohort(fake_explanation: FakeExplanation) -> None:
    with pytest.raises(ValueError, match="both correct and erroneous"):
        shaply.error_analysis(fake_explanation, errors=np.zeros(50, dtype=bool))
