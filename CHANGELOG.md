# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2026-02-05

### Added

- Criterion benchmark infrastructure (bench)

### Changed

- Improve prediction accuracy with cross-validation, dynamic filtering, and softmax-weighted ensemble (trainer)
- Adaptive context length and K selection based on data characteristics (npts)
- Performance optimizations: parallelization, partial sort, vector pre-allocation

## [0.1.4] - 2026-02-01

### Added

- `scaler` crate with `StandardScaler` for z-score normalization
- NPTS model now uses StandardScaler for scale-invariant distance calculation

### Changed

- ETS model falls back to non-seasonal mode when data is insufficient for 2 full cycles (instead of returning an error)

### Fixed

- NPTS predictions now work correctly with extreme value scales (e.g., 1e-9 or 1e12)

## [0.1.3] - 2026-01-30

### Added

- Timestamps field in `ForecastResult` for predictor output

### Fixed

- Predictor now uses ceiling division for horizon to steps conversion

## [0.1.2] - 2026-01-30

### Added

- GitHub Actions CI workflow
- Pre-commit hooks with rusty-hook (fmt + clippy)

### Changed

- Improved `PredictionInput` interface in predictor crate
- Treat warnings as errors via RUSTFLAGS

### Fixed

- Normalize crate handles large time gaps with segment-based normalization

## [0.1.1] - 2026-01-28

### Added

- Initial release with core prediction pipeline
- `common` crate with shared types and error handling
- `normalize` crate for time series normalization
- `analyzer` crate for data analysis
- `selector` crate for model selection
- `models` crate with ETS, ARIMA, and NPTS models
- `trainer` crate for model training
- `predictor` crate for prediction orchestration
- BigDecimal support for external API boundaries
- FFT-based seasonality detection

[Unreleased]: https://github.com/user/zcrc-chronos-rs/compare/0.1.5...HEAD
[0.1.5]: https://github.com/user/zcrc-chronos-rs/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/user/zcrc-chronos-rs/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/user/zcrc-chronos-rs/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/user/zcrc-chronos-rs/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/user/zcrc-chronos-rs/releases/tag/0.1.1
