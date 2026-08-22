# shaply

**Usual SHAP explainability figures, rendered as interactive [Plotly](https://plotly.com/python/) charts.**

`shaply` reproduces the familiar figures from the [`shap`](https://github.com/shap/shap) library
— bar, beeswarm, waterfall, dependence (scatter) and heatmap — but returns
`plotly.graph_objects.Figure` objects instead of matplotlib axes, so the plots are interactive
and embeddable out of the box.

It does **not** depend on `shap`: every plotting function accepts a `shap.Explanation`-like
object, a raw NumPy array of SHAP values, or a pandas `DataFrame`.

## Install

```bash
uv add shaply
# or
pip install shaply
```

## Quickstart

```python
import shaply

fig = shaply.beeswarm(shap_values)  # a shap.Explanation, ndarray or DataFrame
fig.show()
```

Every plotting function accepts an optional typed `config` object (see
[Configuration](api/config.md)) to override defaults such as colors, ordering, or layout.

## Where to go next

- [Usual plots](api/usual.md) — the `shap.plots`-equivalent figures: `bar`, `beeswarm`,
  `waterfall`, `scatter`, `heatmap`, `decision`, `force`.
- [Advanced plots](api/advanced.md) — diagnostics not found in `shap` itself: error analysis,
  monotonicity checks, feature clustering, response curves, and more.
- [Configuration](api/config.md) — the typed `*Config` objects accepted by each plot.
- [Core objects](api/core.md) — `Explanation`, `InteractionValues`, and the coercion helpers.

The full source is on [GitHub](https://github.com/antoine126/shaply); a runnable example
notebook lives in [`examples/`](https://github.com/antoine126/shaply/tree/main/examples).
