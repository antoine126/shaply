from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

FloatArray = npt.NDArray[np.float64]


@dataclass
class FakeExplanation:
    """Minimal stand-in duck-typing ``shap.Explanation``."""

    values: Any
    base_values: Any
    data: Any
    feature_names: Any


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def shap_values(rng: np.random.Generator) -> FloatArray:
    return rng.normal(size=(50, 4)).astype(np.float64)


@pytest.fixture
def feature_data(rng: np.random.Generator) -> FloatArray:
    return rng.normal(size=(50, 4)).astype(np.float64)


@pytest.fixture
def feature_names() -> list[str]:
    return ["age", "income", "score", "tenure"]


@pytest.fixture
def fake_explanation(
    shap_values: FloatArray,
    feature_data: FloatArray,
    feature_names: list[str],
) -> FakeExplanation:
    return FakeExplanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], 0.5),
        data=feature_data,
        feature_names=feature_names,
    )
