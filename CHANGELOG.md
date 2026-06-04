# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Regime detection via Lo-MacKinlay Variance Ratio Test, exposed on
  `TimeSeriesCharacteristics::regime` (analyzer)
- Adaptive linear detrend in the prediction pipeline, gated on
  `regime == Trending` and the absence of log-transform (predictor)
- Direction-based metrics (`DirectionMetrics` with DirAcc, DirAcc
  filtered, IC, calibration residual) and calibration buckets, wired
  into `BacktestResult` and the reporter (bench)
- `ForecastResult.predicted_std`, derived from the 10/90 quantile band
  using `Z_SCORE_80_INTERVAL = 1.282` (predictor)
- `bench::sweep` Cartesian-product evaluator with `run_sweep` driver,
  per-worker `Predictor::new(1)` parallelism, deterministic row sort,
  per-regime / per-series / cross-sectional-IC aggregates, CSV/JSON
  output, JSON-fixture loader, and `predict_sweep` CLI binary (bench)

### Changed

- Adaptive detrend gate switched from the Variance Ratio Test regime
  to a slope-magnitude (span_ratio) gate: detrend is now applied only
  when `|slope| × (n − 1) / |current_price| > 0.15`, with the VR test
  retained as a negative filter (never detrend a MeanReverting series).
  Calibrated against empirical sweeps of NEAR tokens where the VR test
  misclassified stable / bridged tokens with weak persistent drift as
  Trending (predictor)
- `DirectionMetrics.ic` is now `Option<f64>` to distinguish "zero
  variance / no ranking signal" from "true zero correlation" (bench)
- `default_calibration_buckets` outer ranges widened to ±1000% so
  high-volatility meme-coin returns no longer drop silently (bench)
- `bench` crate gains a `cli` feature (default-on) that gates the
  CLI-only dependencies (clap, csv, serde_json, anyhow,
  tracing-subscriber); library consumers may opt out with
  `--no-default-features` (bench)

## [0.1.6] - 2026-03-30

### Added

- Predictor struct with dedicated rayon ThreadPool for concurrent predictions (predictor)
- Memory profiling examples for macOS/Linux (bench)

### Changed

- Upgrade Rust toolchain to 1.94.1 and edition to 2024
- Use array_windows for type-safe slice window access
- Extract safe_exp to module-level function with improved safety (predictor)

### Fixed

- Preserve NaN propagation in safe_exp to prevent silent data corruption (predictor)

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

[Unreleased]: https://github.com/user/zcrc-chronos-rs/compare/0.1.6...HEAD
[0.1.6]: https://github.com/user/zcrc-chronos-rs/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/user/zcrc-chronos-rs/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/user/zcrc-chronos-rs/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/user/zcrc-chronos-rs/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/user/zcrc-chronos-rs/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/user/zcrc-chronos-rs/releases/tag/0.1.1
