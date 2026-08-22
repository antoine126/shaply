"""Cohort-wise feature importance.

Compares ``mean(|SHAP|)`` per feature across cohorts of instances, revealing that
a feature can dominate in one operating regime and be negligible in another.
Cohorts are given explicitly, or built by quantile-binning one feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from shaply.config import ImportanceByCohortConfig
from shaply.explanation import to_explanation
from shaply.plots._common.layout import apply_layout
from shaply.plots._common.ordering import compute_layout, resolve_feature_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from shaply.explanation import Explanation, ExplanationLike

    BoolArray = npt.NDArray[np.bool_]


def importance_by_cohort(
    values: ExplanationLike | npt.ArrayLike | object,
    *,
    cohorts: npt.ArrayLike | None = None,
    by_feature: str | int | None = None,
    base_values: object = None,
    data: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    output_index: int | None = None,
    config: ImportanceByCohortConfig | None = None,
) -> go.Figure:
    """Render feature importance split by cohort.

    Provide the cohorts either directly via ``cohorts`` (one label per instance)
    or via ``by_feature`` (quantile-bin that feature into ``config.n_cohorts``).

    Parameters
    ----------
    values
        SHAP values as an ``Explanation``-like object, numpy array or DataFrame.
    cohorts
        One cohort label per instance.
    by_feature
        Name or index of a feature to quantile-bin into cohorts (needs ``data``).
    base_values, data, feature_names, output_index
        Forwarded to :func:`shaply.explanation.to_explanation`.
    config
        Optional :class:`~shaply.config.ImportanceByCohortConfig`.

    Returns
    -------
    plotly.graph_objects.Figure
        The grouped importance figure.

    Raises
    ------
    ValueError
        If neither or both cohort sources are given, or sizes mismatch.
    """
    explanation = to_explanation(
        values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_index=output_index,
    )
    cfg = config or ImportanceByCohortConfig()
    labels, names = _resolve_cohorts(explanation, cohorts, by_feature, cfg)
    return _build(explanation, cfg, labels, names)


def _resolve_cohorts(
    explanation: Explanation,
    cohorts: npt.ArrayLike | None,
    by_feature: str | int | None,
    cfg: ImportanceByCohortConfig,
) -> tuple[npt.NDArray[np.intp], list[str]]:
    if (cohorts is None) == (by_feature is None):
        msg = "provide exactly one of cohorts=... or by_feature=..."
        raise ValueError(msg)

    if cohorts is not None:
        raw = np.asarray(cohorts)
        if raw.shape != (explanation.n_samples,):
            msg = f"cohorts must have shape ({explanation.n_samples},), got {raw.shape}"
            raise ValueError(msg)
        names = [str(v) for v in np.unique(raw)]
        codes = np.searchsorted(np.unique(raw), raw).astype(np.intp)
        return codes, names

    if explanation.data is None:
        msg = "by_feature requires feature values; pass data=... or a shap.Explanation."
        raise ValueError(msg)
    idx = resolve_feature_index(explanation, by_feature)  # type: ignore[arg-type]
    column = explanation.data[:, idx]
    edges = np.quantile(column, np.linspace(0.0, 1.0, cfg.n_cohorts + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    codes = np.clip(np.digitize(column, edges) - 1, 0, cfg.n_cohorts - 1).astype(np.intp)
    fname = explanation.feature_names[idx]
    names = [f"{fname} Q{c + 1}" for c in range(cfg.n_cohorts)]
    return codes, names


def _build(
    explanation: Explanation,
    cfg: ImportanceByCohortConfig,
    codes: npt.NDArray[np.intp],
    cohort_names: list[str],
) -> go.Figure:
    layout = compute_layout(explanation, cfg.ordering, cfg.max_display)
    order = layout.order[::-1]
    labels = list(layout.labels[::-1])
    abs_values = np.abs(explanation.values)

    fig = go.Figure()
    for c, cohort_name in enumerate(cohort_names):
        mask = codes == c
        if not mask.any():
            continue
        importance = abs_values[mask][:, order].mean(axis=0)
        fig.add_bar(
            x=importance,
            y=labels,
            orientation="h",
            name=f"{cohort_name} (n={int(mask.sum())})",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )

    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.02, "x": 0})
    apply_layout(
        fig,
        cfg,
        title=cfg.title or "Feature importance by cohort",
        xaxis_title="mean(|SHAP value|)",
        yaxis_title=None,
    )
    return fig
