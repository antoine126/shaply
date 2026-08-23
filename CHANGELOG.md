# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2026-08-23

### Added

- Support for Python 3.10 and 3.11. `shaply` now supports Python 3.10
  through 3.13 (previously required `>=3.12.7`).
- `shaply._compat`: internal backport of `enum.StrEnum` (added to the
  standard library in Python 3.11) so `shaply.enums` behaves identically
  on 3.10.

### Changed

- `requires-python` lowered from `>=3.12.7` to `>=3.10`.
- CI test matrix now runs on Python 3.10, 3.11, 3.12 and 3.13 (was 3.12
  and 3.13).
- `ruff`'s `target-version` lowered to `py310` to match.
- Widened the `uv_build` build-backend upper bound from `<0.10.0` to
  `<0.13.0`.

## [1.1.0] - 2026-08-23

### Added

- `scatter_ranges`: new advanced plot - a dependence scatter for one feature
  (real value on x, SHAP value on y) framed by both axes' marginal
  distributions. The real-value box (top) and density silhouette (bottom)
  are gradient-colored on the same low-to-high scale as the points; the
  SHAP-value violin (left) and box (right) stay plain gray.
- `ScatterRangesConfig`.

## [1.0.4] - 2026-08-23

### Added

- README: example `beeswarm_ranges` figure image.

### Changed

- `beeswarm_ranges`: the right-hand "Feature value range" panel no longer uses a
  flat gray `Violin` fill. Each density silhouette is now hand-drawn (own
  Gaussian KDE) and colored with the same low-to-high scale as the beeswarm
  dots, so both panels read as one consistent color language.

### Fixed

- `beeswarm_ranges`: density silhouettes now taper to a clean point exactly at
  each feature's true min/max instead of Plotly's default KDE extrapolation
  overshooting past the data range, which made the min/max value labels hard
  to read against the bulging tail.

## [1.0.3] - 2026-08-22

### Added

- Documentation site built with `mkdocs-material` and `mkdocstrings`, with an
  API reference auto-generated from the existing numpy-style docstrings
  (usual plots, advanced plots, configuration objects, core objects).
- GitHub Actions workflow deploying the documentation to GitHub Pages on
  every push to `main`.
- `Documentation` and `Issues` links in `pyproject.toml`'s `project.urls`,
  and a corresponding badge in the README.

## [1.0.2] - 2026-08-22

### Added

- README badges: test status, Codecov coverage, PyPI version, supported
  Python versions, PyPI downloads, and license.
- GitHub Actions workflow running the test suite (Python 3.12 and 3.13) with
  coverage uploaded to Codecov.

## [1.0.1] - 2026-08-22

### Changed

- `Development Status` classifier bumped from `3 - Alpha` to
  `5 - Production/Stable`.
- Added `Intended Audience :: Developers`, `Operating System :: OS Independent`,
  and `Topic :: Software Development :: Libraries :: Python Modules`
  classifiers.

## [1.0.0] - 2026-08-22

### Added

- Initial release.
- Seven "usual" SHAP-equivalent Plotly figures: `bar`, `beeswarm`, `waterfall`,
  `scatter`, `heatmap`, `force`, `decision`.
- Ten advanced diagnostic figures: `beeswarm_ranges`, `response_curve`,
  `interaction_heatmap`, `error_analysis`, `shap_surface`,
  `importance_by_cohort`, `feature_clustering`, `explanation_archetypes`,
  `importance_ci`, `monotonicity_check`.
- No hard dependency on `shap`: every function accepts a `shap.Explanation`-like
  object, a raw NumPy array, or a pandas `DataFrame`.
- Typed, validated configuration objects (Pydantic v2) for every plot.

[Unreleased]: https://github.com/antoine126/shaply/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/antoine126/shaply/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/antoine126/shaply/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/antoine126/shaply/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/antoine126/shaply/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/antoine126/shaply/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/antoine126/shaply/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/antoine126/shaply/releases/tag/v1.0.0
