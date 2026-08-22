from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import shaply
from shaply.config import (
    ExplanationArchetypesConfig,
    ImportanceByCohortConfig,
    ImportanceCIConfig,
)
from shaply.plots._common.cluster import average_linkage_order, kmeans
from shaply.plots._common.stats import average_ranks, spearman
from tests.conftest import FakeExplanation


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def test_average_ranks_handles_ties() -> None:
    ranks = average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    assert ranks.tolist() == [0.0, 1.5, 1.5, 3.0]


def test_spearman_monotonic_is_one() -> None:
    x = np.arange(20, dtype=float)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_average_linkage_order_is_permutation() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(6, 3))
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    order = average_linkage_order(dist)
    assert sorted(order) == list(range(6))


def test_kmeans_separates_clusters() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.3, size=(40, 2))
    b = rng.normal(5, 0.3, size=(40, 2))
    labels, centers = kmeans(np.vstack([a, b]), 2, random_state=0)
    assert centers.shape == (2, 2)
    # The two halves should each be (almost) pure single clusters.
    assert len(set(labels[:40])) == 1
    assert len(set(labels[40:])) == 1
    assert labels[0] != labels[-1]


# --------------------------------------------------------------------------- #
# importance_ci
# --------------------------------------------------------------------------- #
def test_importance_ci_has_error_bars(shap_values: np.ndarray) -> None:
    fig = shaply.importance_ci(shap_values, config=ImportanceCIConfig(n_boot=100))
    assert isinstance(fig, go.Figure)
    assert fig.data[0].error_x.array is not None


def test_importance_ci_is_deterministic(shap_values: np.ndarray) -> None:
    cfg = ImportanceCIConfig(n_boot=100, random_state=3)
    a = shaply.importance_ci(shap_values, config=cfg).data[0].error_x.array
    b = shaply.importance_ci(shap_values, config=cfg).data[0].error_x.array
    assert np.allclose(a, b)


# --------------------------------------------------------------------------- #
# monotonicity_check
# --------------------------------------------------------------------------- #
def test_monotonicity_requires_data(shap_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="requires feature values"):
        shaply.monotonicity_check(shap_values)


def test_monotonicity_detects_increasing() -> None:
    x = np.linspace(0, 1, 200)
    sv = np.column_stack([x, np.zeros_like(x)])  # feature 0 SHAP == its value
    data = np.column_stack([x, np.random.default_rng(0).normal(size=x.size)])
    fig = shaply.monotonicity_check(sv, data=data, feature_names=["mono", "flat"])
    # Top bar (last in the reversed list) is the perfectly monotonic feature.
    assert fig.data[0].y[-1] == "mono"
    assert fig.data[0].x[-1] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# importance_by_cohort
# --------------------------------------------------------------------------- #
def test_importance_by_cohort_explicit(fake_explanation: FakeExplanation) -> None:
    cohorts = np.where(np.arange(50) < 25, "A", "B")
    fig = shaply.importance_by_cohort(fake_explanation, cohorts=cohorts)
    names = [tr.name for tr in fig.data]
    assert any("A" in n for n in names) and any("B" in n for n in names)


def test_importance_by_cohort_by_feature(fake_explanation: FakeExplanation) -> None:
    fig = shaply.importance_by_cohort(
        fake_explanation, by_feature="age", config=ImportanceByCohortConfig(n_cohorts=3)
    )
    assert len(fig.data) == 3


def test_importance_by_cohort_rejects_both(fake_explanation: FakeExplanation) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        shaply.importance_by_cohort(fake_explanation, cohorts=np.zeros(50), by_feature="age")


# --------------------------------------------------------------------------- #
# shap_surface
# --------------------------------------------------------------------------- #
def test_shap_surface_requires_data(shap_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="requires feature values"):
        shaply.shap_surface(shap_values, 0, 1)


def test_shap_surface_grid_shape(fake_explanation: FakeExplanation) -> None:
    fig = shaply.shap_surface(fake_explanation, "age", "income")
    assert fig.data[0].type == "heatmap"
    assert np.asarray(fig.data[0].z).shape == (20, 20)


# --------------------------------------------------------------------------- #
# feature_clustering
# --------------------------------------------------------------------------- #
def test_feature_clustering_diagonal_is_one() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(100, 1))
    # f1 == f2 (redundant), f3 independent.
    sv = np.column_stack([base, base + rng.normal(0, 0.01, (100, 1)), rng.normal(size=(100, 1))])
    fig = shaply.feature_clustering(sv, feature_names=["f1", "f2", "f3"])
    z = np.asarray(fig.data[0].z, dtype=float)
    assert np.allclose(np.diag(z), 1.0)
    assert z.shape == (3, 3)


# --------------------------------------------------------------------------- #
# explanation_archetypes
# --------------------------------------------------------------------------- #
def test_explanation_archetypes_rows(fake_explanation: FakeExplanation) -> None:
    fig = shaply.explanation_archetypes(
        fake_explanation, config=ExplanationArchetypesConfig(n_clusters=3, max_display=4)
    )
    z = np.asarray(fig.data[0].z)
    assert z.shape[0] <= 3  # up to n_clusters archetype rows
    assert z.shape[1] == 4  # displayed features
