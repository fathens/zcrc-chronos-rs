# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `ThetaModel` now damps its theta=0 (linear) extrapolation:
  `slope × Σᵢ₌₁ʰ⁺¹ φⁱ` with `THETA_DAMPING_PHI = 0.97` replaces the
  earlier undamped `slope × (n + h)`. The damped sum asymptotes at
  `slope × φ / (1 − φ) ≈ 32` so a one-week-ahead forecast carries
  about 5× less linear extrapolation than the previous implementation.
  Production diagnostic
  (`real_data_over_damping::top_decile_decomposition`, 1084 NEAR-token
  snapshots at h = 168) showed Theta as the most frequent top-decile
  puller (47/108), driving the "high-confidence wrong-direction" tail
  via single-period linear extrapolation. `φ` is `pub(crate)` so the
  production team can grid-search it (models)
- `NptsModel` now trims the K-neighbour subsequent-value slate
  per-horizon-step before the inverse-distance weighted mean: the
  largest and smallest `NPTS_TRIM_PER_END = 1` neighbour value at
  each step are dropped, the rest are weighted as before. NPTS
  supplied 7/15 of the top-magnitude predictions in the production
  diagnostic, including +10–35 % outliers whose realised returns
  were flat to slightly negative; the trim makes the forecast robust
  to neighbours that happen to precede a single rally without
  discarding the inverse-distance weighting on the rest of the K
  slate. `NPTS_TRIM_PER_END` is `pub(crate)` so the production team
  can grid-search it (models)
- Non-seasonal augurs `AutoETS` spec is now `"ZAN"` instead of `"ZZN"`
  in `EtsModel`, `MstlEtsModel` (no-period fallback and trend model),
  and the theta-2 line of `ThetaModel`. The previous `"ZZN"` let the
  AICc search pick `ETS(A,N,N)` — "no trend" — which fits a flat line
  for the entire forecast horizon. On low-SNR series the AICc penalty
  for the additional trend parameter routinely won out, and the
  `real_data_over_damping` diagnostic measured EtsModel and
  MstlEtsModel returning flat predictions on 56 % of the existing real
  fixtures (FullPipeline 37 %). Forcing the trend component to
  Additive (still damped/undamped selected by AICc) collapses the
  per-model flat rate to 16 % across the same fixtures, with the
  trend coefficient free to shrink toward zero on genuinely flat
  series. This addresses the production "168h flat 79 %" symptom that
  the earlier seasonality fix alone could not move (models)
- `TimeSeriesAnalyzer::detect_seasonality` no longer reports a numeric
  `period` when the spectral peak is classified as "weak" (score ≤
  0.1). The FFT always returns *some* peak on white-noise data, and
  forwarding that spurious period through `TrainingHints` sent
  EtsModel down the seasonal Holt-Winters path. HW fits trend ≈ 0
  with near-constant seasonal indices on noise, collapsing
  multi-step forecasts to a flat line. The "flat" and "outlier"
  fixtures in the `analyzer_output.json` golden now report
  `period: null` to match (analyzer)

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
- Per-horizon sweep aggregation (`HorizonStats`, `aggregate_by_horizon`):
  flat-prediction count, average DirAcc / per-row IC, cross-sectional
  IC across the rows at each horizon, and decile spread of actual
  returns ranked by predicted returns (bench)

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
