"""Color utilities reproducing SHAP's visual language for Plotly figures.

SHAP relies on a red/blue diverging scheme where blue encodes low feature
values and red encodes high ones. The exact anchor colors used here match the
defaults shipped by the ``shap`` package so figures feel familiar to its users.
"""

from __future__ import annotations

from typing import Final

from shaply.enums import ColorScale

#: Canonical SHAP accent colors.
SHAP_RED: Final = "#ff0d57"
SHAP_BLUE: Final = "#008bff"
SHAP_GRAY: Final = "#777777"

#: Plotly colorscale definitions as ``(position, css_color)`` stops.
_ColorStop = tuple[float, str]
Colorscale = list[_ColorStop]

_RED_BLUE: Final[Colorscale] = [
    (0.0, SHAP_BLUE),
    (0.5, "#c6b7d6"),
    (1.0, SHAP_RED),
]

_COOLWARM: Final[Colorscale] = [
    (0.0, "#3b4cc0"),
    (0.5, "#dddddd"),
    (1.0, "#b40426"),
]

#: Sequential white->red scale for non-negative magnitudes (e.g. |interaction|).
_REDS: Final[Colorscale] = [
    (0.0, "#fff5f0"),
    (0.5, "#fca082"),
    (1.0, SHAP_RED),
]

_SCALE_REGISTRY: Final[dict[ColorScale, Colorscale | str]] = {
    ColorScale.RED_BLUE: _RED_BLUE,
    ColorScale.COOLWARM: _COOLWARM,
    ColorScale.REDS: _REDS,
    ColorScale.VIRIDIS: "Viridis",
    ColorScale.PLASMA: "Plasma",
}


def resolve_colorscale(scale: ColorScale) -> Colorscale | str:
    """Return the Plotly-compatible colorscale for a :class:`ColorScale`.

    Parameters
    ----------
    scale
        The named color scale to resolve.

    Returns
    -------
    list of tuple or str
        Either an explicit list of ``(position, color)`` stops or the name of a
        built-in Plotly colorscale.
    """
    return _SCALE_REGISTRY[scale]


def sample_scale(scale: ColorScale, positions: list[float]) -> list[str]:
    """Sample a color scale at ``positions`` in ``[0, 1]``.

    Parameters
    ----------
    scale
        The named color scale to sample.
    positions
        Fractions in ``[0, 1]`` at which to read the scale.

    Returns
    -------
    list of str
        One CSS ``rgb(...)`` color per requested position.
    """
    from plotly.colors import sample_colorscale

    resolved = resolve_colorscale(scale)
    colorscale = resolved if isinstance(resolved, str) else [list(stop) for stop in resolved]
    return list(sample_colorscale(colorscale, positions, colortype="rgb"))


def signed_color(value: float) -> str:
    """Return the SHAP accent color matching the sign of ``value``.

    Positive contributions map to red, negative ones to blue, matching the
    waterfall and force conventions of ``shap``.
    """
    return SHAP_RED if value >= 0 else SHAP_BLUE
