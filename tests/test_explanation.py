from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shaply.explanation import Explanation, to_explanation
from tests.conftest import FakeExplanation


def test_to_explanation_from_ndarray_defaults(shap_values: np.ndarray) -> None:
    exp = to_explanation(shap_values)
    assert exp.n_samples == 50
    assert exp.n_features == 4
    assert exp.feature_names == ("Feature 0", "Feature 1", "Feature 2", "Feature 3")
    assert np.allclose(exp.base_values, 0.0)
    assert exp.data is None


def test_to_explanation_promotes_1d() -> None:
    exp = to_explanation(np.array([1.0, -2.0, 3.0]))
    assert exp.values.shape == (1, 3)
    assert exp.is_single


def test_to_explanation_from_dataframe() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    exp = to_explanation(frame)
    assert exp.feature_names == ("a", "b")
    assert exp.values.shape == (2, 2)


def test_to_explanation_from_shap_like(fake_explanation: FakeExplanation) -> None:
    exp = to_explanation(fake_explanation)
    assert exp.feature_names == ("age", "income", "score", "tenure")
    assert exp.data is not None
    assert np.allclose(exp.base_values, 0.5)


def test_base_values_scalar_broadcast(shap_values: np.ndarray) -> None:
    exp = to_explanation(shap_values, base_values=1.5)
    assert exp.base_values.shape == (50,)
    assert np.allclose(exp.base_values, 1.5)


def test_multi_output_requires_index() -> None:
    values = np.zeros((10, 4, 3))
    with pytest.raises(ValueError, match="output_index"):
        to_explanation(values)
    exp = to_explanation(values, output_index=1)
    assert exp.values.shape == (10, 4)


def test_multi_output_explanation_selects_class_base() -> None:
    # Mimics a binary classifier shap.Explanation: 3D values, 2D base_values.
    values = np.arange(10 * 4 * 2, dtype=float).reshape(10, 4, 2)
    base = np.stack([np.zeros(10), np.ones(10)], axis=1)  # (10, 2)
    exp = to_explanation(
        FakeExplanation(values=values, base_values=base, data=None, feature_names=None),
        output_index=1,
    )
    assert exp.values.shape == (10, 4)
    assert np.allclose(exp.values, values[..., 1])
    assert np.allclose(exp.base_values, 1.0)


def test_inconsistent_feature_names_raise(shap_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="feature_names must have length"):
        to_explanation(shap_values, feature_names=["only_one"])


def test_mean_abs_and_select_sample(shap_values: np.ndarray) -> None:
    exp = to_explanation(shap_values)
    assert exp.mean_abs().shape == (4,)
    single = exp.select_sample(3)
    assert single.is_single
    assert np.allclose(single.values[0], shap_values[3])


def test_explanation_validates_base_shape() -> None:
    with pytest.raises(ValueError, match="base_values must have shape"):
        Explanation(
            values=np.zeros((3, 2)),
            base_values=np.zeros(2),
            feature_names=("a", "b"),
        )
