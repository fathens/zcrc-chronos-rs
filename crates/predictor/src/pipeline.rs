use std::collections::BTreeMap;
use std::sync::LazyLock;

use analyzer::TimeSeriesAnalyzer;
use chrono::{NaiveDateTime, TimeDelta};
use common::{
    BigDecimal, ChronosError, Result, TimeSeriesCharacteristics, TimeSeriesRegime,
    decimals_to_f64s, f64s_to_decimals,
};
use normalize::normalize_time_series_data;
use selector::AdaptiveModelSelector;
use serde::{Deserialize, Serialize};
use tracing::info;
use trainer::HierarchicalTrainer;

/// Input for the prediction pipeline.
#[derive(Debug, Clone)]
pub struct PredictionInput {
    /// Time-series data as timestamp → value mapping.
    /// BTreeMap ensures data is always sorted by timestamp.
    pub data: BTreeMap<NaiveDateTime, BigDecimal>,

    /// Forecast horizon as a duration (e.g., 24 hours ahead).
    pub horizon: TimeDelta,
}

/// Z-score for the 80% prediction interval (10th / 90th percentiles).
/// Used to derive `predicted_std` from the quantile band as
/// `(upper - lower) / (2 * Z_SCORE_80_INTERVAL)`.
pub(crate) const Z_SCORE_80_INTERVAL: f64 = 1.281_551_565_544_6;

/// Minimum span_ratio (= `|slope| × (n − 1) / |current_price|`) required
/// to apply the linear detrend pre-processor. The ratio expresses the
/// relative price change that the linear trend implies over the full
/// training window — at 0.15, the trend would account for ≥15 % of the
/// current price.
///
/// Calibrated against an empirical sweep of 271 NEAR tokens (`improve.md`,
/// 2026-06-04 verification): θ = 0.15 cleanly excludes stable / bridged
/// tokens (whose autocorrelation-significant drifts look "Trending" to the
/// Variance Ratio Test but carry essentially no economic trend) while still
/// catching the strong-trend periods of the trending winners that motivated
/// detrend in the first place.
pub(crate) const DETREND_SPAN_RATIO_THRESHOLD: f64 = 0.15;

/// Floor on `|current_price|` used as the denominator of the span_ratio
/// gate. Anything below this is treated as an unreliable baseline — for
/// price series that have collapsed close to zero we keep the raw forecast
/// rather than risk dividing by a near-singular value.
const DETREND_PRICE_FLOOR: f64 = 1e-150;

/// Full result of a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Point forecast values with timestamps.
    pub forecast_values: BTreeMap<NaiveDateTime, BigDecimal>,
    /// Lower confidence bound (10th percentile) with timestamps.
    pub lower_bound: Option<BTreeMap<NaiveDateTime, BigDecimal>>,
    /// Upper confidence bound (90th percentile) with timestamps.
    pub upper_bound: Option<BTreeMap<NaiveDateTime, BigDecimal>>,
    /// Approximated per-step predicted standard deviation in the original
    /// (output) scale, derived from the 10/90 quantile band as
    /// `(upper - lower) / (2 * Z_SCORE_80_INTERVAL)`. Assumes the band is
    /// approximately Gaussian; for log-transformed series this is a
    /// first-order approximation in the exponentiated domain. `None` when
    /// either quantile bound is unavailable.
    pub predicted_std: Option<BTreeMap<NaiveDateTime, BigDecimal>>,
    /// Name of model(s) used.
    pub model_name: String,
    /// Strategy selected.
    pub strategy_name: String,
    /// Total processing time in seconds.
    pub processing_time_secs: f64,
    /// Number of models trained.
    pub model_count: usize,
}

/// Prediction engine with a dedicated thread pool for model training.
///
/// Model training runs on pool threads via `pool.spawn()`, ensuring the calling
/// thread (e.g., `tokio::spawn_blocking`) does NOT participate in rayon's
/// work-stealing. This guarantees that the number of concurrent model trainings
/// equals `pool_threads`, regardless of how many predictions run concurrently.
pub struct Predictor {
    pool: rayon::ThreadPool,
}

impl Predictor {
    /// Create a new predictor with a dedicated thread pool.
    ///
    /// `max_model_threads` controls peak memory: each thread can hold one
    /// augurs model buffer (~200 MB). Recommended: 3 for balance of speed
    /// and memory (~856 MiB total, ~2 min for 80 tokens).
    pub fn new(max_model_threads: usize) -> Result<Self> {
        if max_model_threads == 0 {
            return Err(ChronosError::InvalidInput(
                "max_model_threads must be > 0".into(),
            ));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_model_threads)
            .build()
            .map_err(|e| ChronosError::ModelError(format!("ThreadPool creation failed: {e}")))?;
        Ok(Self { pool })
    }

    /// Run the prediction pipeline.
    ///
    /// Pipeline phases:
    /// 1. Pre-training (calling thread): normalize → analyze → strategy selection
    /// 2. Training (pool threads): hierarchical model training + ensemble
    /// 3. Post-training (calling thread): inverse transform → decimal conversion
    pub fn predict(&self, input: &PredictionInput) -> Result<ForecastResult> {
        let start = std::time::Instant::now();

        // Validate input
        if input.data.len() < 2 {
            return Err(ChronosError::InsufficientData(
                "At least 2 data points required".into(),
            ));
        }
        if input.horizon <= TimeDelta::zero() {
            return Err(ChronosError::InvalidInput(
                "horizon must be positive".into(),
            ));
        }

        // --- Phase 1: Pre-training (calling thread) ---

        // Extract sorted data from BTreeMap
        let (timestamps, decimal_values): (Vec<_>, Vec<_>) = input
            .data
            .iter()
            .map(|(ts, val)| (*ts, val.clone()))
            .unzip();

        // Convert Decimal → f64 at the boundary (NaN/Inf cannot exist in Decimal)
        let f64_values = decimals_to_f64s(&decimal_values)?;

        // Auto-calculate time budget based on data size
        let time_budget = calculate_time_budget(input.data.len());

        info!(
            data_points = input.data.len(),
            horizon = %input.horizon,
            time_budget_secs = time_budget,
            "Starting prediction pipeline"
        );

        // Step 1: Normalize
        let (norm_timestamps, norm_values) = normalize_time_series_data(&timestamps, &f64_values)?;

        // Step 2: Analyze characteristics (run once, share with selector and trainer)
        let analyzer = TimeSeriesAnalyzer::new();
        let characteristics = analyzer.analyze(&norm_values, &norm_timestamps);
        let season_period = characteristics.seasonality.period;
        let is_exponential = characteristics.trend.is_exponential;

        // Step 2.5: Apply log transform if exponential trend detected
        let (log_values, log_transformed) = if is_exponential {
            if norm_values.iter().all(|&v| v > 0.0) {
                info!("Exponential trend detected, applying log transform");
                let log_vals: Vec<f64> = norm_values.iter().map(|v| v.ln()).collect();
                (log_vals, true)
            } else {
                info!(
                    "Exponential trend detected but data contains non-positive values, skipping log transform"
                );
                (norm_values.clone(), false)
            }
        } else {
            (norm_values.clone(), false)
        };

        // Step 2.6: Apply linear detrend when the linear trend would account
        // for ≥`DETREND_SPAN_RATIO_THRESHOLD` of the current price over the
        // training window. Autocorrelation-based regime detection (Variance
        // Ratio Test) cannot tell economically meaningful trend from a
        // mean-reverting series with weak persistent drift — the latter
        // includes stable / bridged crypto tokens that look "Trending" to
        // the test but should not be detrended.
        //
        // VR test is used as a negative filter only: a series that is
        // statistically MeanReverting will not be detrended regardless of
        // its slope. Log-transformed series skip the gate because the log
        // already compresses exponential trends.
        let detrend_state = compute_detrend_state(&characteristics, &log_values, log_transformed);

        // Snapshot the same-domain history before `train_values` consumes
        // `log_values`. The Step 5a.5 reverting blend needs the
        // trend-included series (log when log_transformed, raw price
        // otherwise) so the gate's pred_return and the AR(1) fit share
        // the domain that `mean_after_trend` is computed in.
        let revert_fit_values: Vec<f64> = log_values.clone();

        let train_values: Vec<f64> = if let Some(ref state) = detrend_state {
            log_values
                .iter()
                .enumerate()
                .map(|(i, &v)| v - (state.slope * i as f64 + state.intercept))
                .collect()
        } else {
            log_values
        };

        // Step 3: Select strategy from pre-computed characteristics
        let selector = AdaptiveModelSelector::default();
        let strategy = selector.select_strategy_from_characteristics(
            &characteristics,
            train_values.len(),
            time_budget as u64,
        );

        // Convert horizon from TimeDelta to steps with timestamps
        let (horizon_steps, forecast_timestamps) =
            horizon_to_steps_with_timestamps(&input.horizon, &norm_timestamps);

        // Extract strategy_name before moving strategy into the pool closure
        let strategy_name = strategy.strategy_name.clone();

        // --- Phase 2: Training (pool threads) ---
        // pool.spawn (not pool.install): ensures the calling thread does NOT
        // participate in rayon's work-stealing, so concurrent predict() calls
        // each wait on their own channel without becoming worker threads.

        let hints = trainer::TrainingHints {
            season_period,
            volatility: Some(characteristics.volatility),
        };

        let (forecast, metadata) = {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);

            self.pool.spawn(move || {
                let mut trainer = HierarchicalTrainer::default();
                let result = trainer.train_hierarchically(
                    &train_values,
                    &norm_timestamps,
                    &strategy,
                    time_budget,
                    horizon_steps,
                    hints,
                );
                if tx.send(result).is_err() {
                    tracing::warn!("prediction result channel closed; caller may have panicked");
                }
            });

            rx.recv()
                .map_err(|_| ChronosError::ModelError("model training thread panicked".into()))?
        }?;

        // --- Phase 3: Post-training (calling thread) ---

        // Step 5a: Add the linear trend back to the forecast if detrend was applied.
        // Forecast step i corresponds to original index (training_len + i).
        // The damping coefficient `phi` (default 1.0 = undamped) is read
        // once per `predict()` invocation to keep zaciraci's predict_sweep
        // grid-search runs internally consistent.
        let phi = retrend_damping_phi();
        let mean_after_trend = add_trend_if_detrended(forecast.mean, &detrend_state, phi);
        let lower_after_trend = forecast
            .lower_quantile
            .map(|v| add_trend_if_detrended(v, &detrend_state, phi));
        let upper_after_trend = forecast
            .upper_quantile
            .map(|v| add_trend_if_detrended(v, &detrend_state, phi));

        // Step 5a.5: Reverting blend post-process. Disabled by default
        // (`CHRONOS_REVERT_BLEND_ALPHA` unset → α = 0). When enabled,
        // the overlay shrinks confident momentum forecasts toward an
        // explicit AR(1) reverting prediction so the ensemble can
        // escape the convex hull of its momentum-only base models on
        // mean-reverting forward returns. All gate / domain decisions
        // live in `revert_blend::apply_revert_blend`; the pipeline
        // only feeds env-derived thresholds and the same-domain
        // anchor / history.
        let revert_alpha = revert_blend_alpha();
        let revert_theta = revert_magnitude_threshold();
        let revert_horizon_min = revert_horizon_min_secs();
        let horizon_secs = input.horizon.num_seconds().max(0) as u64;
        let revert_eligible = revert_alpha > 0.0
            && horizon_secs >= revert_horizon_min
            && !revert_fit_values.is_empty();

        let (mean_after_trend, lower_after_trend, upper_after_trend, blend_applied) =
            if revert_eligible {
                let anchor = *revert_fit_values
                    .last()
                    .expect("revert_eligible verified non-empty above");
                // `Ar1RevertingModel::fit_predict` ignores timestamps,
                // so an empty slice is sufficient here and avoids
                // cloning `norm_timestamps` (which has already been
                // moved into the training pool closure above).
                match revert_blend::apply_revert_blend(
                    &mean_after_trend,
                    lower_after_trend.as_deref(),
                    upper_after_trend.as_deref(),
                    &revert_fit_values,
                    &[],
                    anchor,
                    log_transformed,
                    revert_alpha,
                    revert_theta,
                ) {
                    Some(blended) => {
                        info!(
                            alpha = revert_alpha,
                            magnitude_threshold = revert_theta,
                            horizon_min_secs = revert_horizon_min,
                            pred_return = blended.pred_return,
                            "reverting blend applied"
                        );
                        (blended.mean, blended.lower, blended.upper, true)
                    }
                    None => (
                        mean_after_trend,
                        lower_after_trend,
                        upper_after_trend,
                        false,
                    ),
                }
            } else {
                (
                    mean_after_trend,
                    lower_after_trend,
                    upper_after_trend,
                    false,
                )
            };

        // Step 5b: Inverse transform if log was applied
        let final_mean: Vec<f64> = if log_transformed {
            mean_after_trend.iter().copied().map(safe_exp).collect()
        } else {
            mean_after_trend
        };

        let final_lower: Option<Vec<f64>> = if log_transformed {
            lower_after_trend.map(|v| v.iter().copied().map(safe_exp).collect())
        } else {
            lower_after_trend
        };

        let final_upper: Option<Vec<f64>> = if log_transformed {
            upper_after_trend.map(|v| v.iter().copied().map(safe_exp).collect())
        } else {
            upper_after_trend
        };

        let processing_time = start.elapsed().as_secs_f64();

        info!(
            strategy = %strategy_name,
            models = metadata.model_count,
            log_transformed = log_transformed,
            detrended = detrend_state.is_some(),
            time_secs = processing_time,
            "Prediction pipeline complete"
        );

        // Convert f64 → Decimal and build BTreeMap with timestamps
        let forecast_decimals = f64s_to_decimals(&final_mean)?;
        let forecast_values: BTreeMap<NaiveDateTime, BigDecimal> = forecast_timestamps
            .iter()
            .zip(forecast_decimals)
            .map(|(ts, val)| (*ts, val))
            .collect();

        let lower_bound = final_lower
            .as_ref()
            .map(|v| {
                f64s_to_decimals(v).map(|decimals| {
                    forecast_timestamps
                        .iter()
                        .zip(decimals.into_iter())
                        .map(|(ts, val)| (*ts, val))
                        .collect()
                })
            })
            .transpose()?;

        let upper_bound = final_upper
            .as_ref()
            .map(|v| {
                f64s_to_decimals(v).map(|decimals| {
                    forecast_timestamps
                        .iter()
                        .zip(decimals.into_iter())
                        .map(|(ts, val)| (*ts, val))
                        .collect()
                })
            })
            .transpose()?;

        let predicted_std = match (final_lower.as_ref(), final_upper.as_ref()) {
            (Some(lo), Some(hi)) if lo.len() == hi.len() => {
                let std_values: Vec<f64> = lo
                    .iter()
                    .zip(hi.iter())
                    .map(|(l, u)| ((u - l) / (2.0 * Z_SCORE_80_INTERVAL)).max(0.0))
                    .collect();
                let decimals = f64s_to_decimals(&std_values)?;
                let map: BTreeMap<NaiveDateTime, BigDecimal> = forecast_timestamps
                    .iter()
                    .zip(decimals)
                    .map(|(ts, val)| (*ts, val))
                    .collect();
                Some(map)
            }
            _ => None,
        };

        let model_name = if blend_applied {
            let mut s = forecast.model_name;
            s.push_str("+Ar1Revert");
            s
        } else {
            forecast.model_name
        };

        Ok(ForecastResult {
            forecast_values,
            lower_bound,
            upper_bound,
            predicted_std,
            model_name,
            strategy_name,
            processing_time_secs: processing_time,
            model_count: metadata.model_count,
        })
    }
}

/// State captured when linear detrend is applied, used to re-add the trend
/// to the forecast in post-processing.
#[derive(Debug, Clone)]
struct DetrendState {
    slope: f64,
    intercept: f64,
    /// Number of training samples (the index where forecasting begins).
    training_len: usize,
}

/// Environment variable controlling whether the adaptive linear detrend
/// stage runs at all. Setting it to `0`, `false`, or `off`
/// (case-insensitive) bypasses both `compute_detrend_state` and the
/// re-trend step, so the predictor returns raw model output without
/// adding back any extrapolated linear trend.
///
/// Default behaviour (variable unset, or set to anything else) is
/// **detrend enabled** — identical to the pre-flag implementation.
///
/// Exposed as a runtime knob because the zaciraci 06-12 verification
/// showed that on 38 % of production TOP-decile rows the
/// FullPipeline forecast exceeded every individual model, with the
/// excess traceable to the linear retrend extrapolation over
/// `h = 168` h. Toggling this variable lets the production team A/B
/// the entire detrend stage in `predict_sweep` without recompiling.
const ADAPTIVE_DETREND_ENV: &str = "CHRONOS_ADAPTIVE_DETREND";

/// Environment variable controlling the damping coefficient applied to
/// the re-trend extrapolation in [`add_trend_if_detrended`]. The
/// default is [`RETREND_DAMPING_PHI_DEFAULT`]. Values in `(0, 1)`
/// replace the per-step `i` term with the geometric sum
/// `Σⱼ₌₁ⁱ φʲ = φ × (1 - φⁱ) / (1 - φ)`, which asymptotes at
/// `φ / (1 - φ)` and so bounds the long-horizon extrapolation;
/// `φ = 1.0` reproduces the original undamped behaviour
/// `slope × (training_len + i)`.
///
/// Values outside `(0, 1]`, NaN, and unparseable strings are ignored
/// (the default is used).
const RETREND_DAMPING_PHI_ENV: &str = "CHRONOS_RETREND_DAMPING_PHI";

/// Default damping coefficient used when [`RETREND_DAMPING_PHI_ENV`]
/// is unset or unparseable. Production `predict_sweep` on the
/// 271-token / 27,100-prediction NEAR universe identified `φ = 0.90`
/// as the empirical optimum: the IC at the production `h = 168`
/// horizon flipped from `-0.031` (undamped) to `+0.012`, and the
/// decile spread improved from `-4.82 %` to `-0.59 %`. The
/// `0.85 / 0.90` plateau pins this to the recommended grid endpoint
/// without overshooting into territory where the retrend stops
/// adding any directional information.
pub(crate) const RETREND_DAMPING_PHI_DEFAULT: f64 = 0.90;

/// Parse the [`ADAPTIVE_DETREND_ENV`] payload. Pulled out of the
/// env-reading helper so that callers (and tests) can exercise the
/// truthiness rules without mutating process state.
fn parse_adaptive_detrend_value(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            !(v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no"))
        }
        None => true,
    }
}

/// Read [`ADAPTIVE_DETREND_ENV`] and return whether the detrend stage
/// should run. Defaults to `true` when the variable is unset.
fn adaptive_detrend_enabled() -> bool {
    parse_adaptive_detrend_value(std::env::var(ADAPTIVE_DETREND_ENV).ok().as_deref())
}

/// Parse the [`RETREND_DAMPING_PHI_ENV`] payload, falling back to
/// [`RETREND_DAMPING_PHI_DEFAULT`] on any malformed or out-of-range
/// input.
fn parse_retrend_damping_phi(value: Option<&str>) -> f64 {
    value
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v.is_finite() && v > 0.0 && v <= 1.0)
        .unwrap_or(RETREND_DAMPING_PHI_DEFAULT)
}

/// Read [`RETREND_DAMPING_PHI_ENV`] and return a valid damping factor.
fn retrend_damping_phi() -> f64 {
    parse_retrend_damping_phi(std::env::var(RETREND_DAMPING_PHI_ENV).ok().as_deref())
}

/// Environment variable: minimum forecast horizon (in seconds) for the
/// reverting blend post-process to fire. Horizons below this threshold
/// skip the AR(1) overlay entirely, regardless of the predicted return
/// magnitude. Default `0` lets every horizon trigger when the
/// magnitude gate is met; raising this — e.g. to `604_800` (one week)
/// — restricts the overlay to long-horizon predictions where the
/// momentum-vs-reversion cross-section bites hardest.
const REVERT_HORIZON_MIN_SECS_ENV: &str = "CHRONOS_REVERT_HORIZON_MIN_SECS";

/// Environment variable: simple-return magnitude threshold for the
/// reverting blend gate. Predictions whose `|pred_return|` is at or
/// below this value pass through unchanged. Default `0.05` (5 %)
/// targets the "confident momentum" tail that `improve.md` 2026-06-15
/// updates 7-8 identified as the source of the confident-wrong loss.
const REVERT_MAGNITUDE_THRESHOLD_ENV: &str = "CHRONOS_REVERT_MAGNITUDE_THRESHOLD";

/// Environment variable: per-step blend weight on the AR(1) reverting
/// forecast. Acceptable range `(0, 1]`. `α = 0` (default) disables the
/// overlay completely — the rest of the pipeline returns exactly the
/// pre-Step-2 forecast. `α = 1` replaces the base forecast with the
/// AR(1) forecast outright. Intermediate values produce a per-step
/// convex combination `(1 − α) · base + α · ar1`.
const REVERT_BLEND_ALPHA_ENV: &str = "CHRONOS_REVERT_BLEND_ALPHA";

/// Default for [`REVERT_HORIZON_MIN_SECS_ENV`]. `0` keeps every
/// horizon eligible for the overlay so the magnitude gate alone
/// decides whether it fires.
pub(crate) const REVERT_HORIZON_MIN_SECS_DEFAULT: u64 = 0;

/// Default for [`REVERT_MAGNITUDE_THRESHOLD_ENV`]. Five percent picks
/// up the confident-wrong tail that production tracked at
/// `pred > +5 %` while leaving small-magnitude forecasts untouched.
pub(crate) const REVERT_MAGNITUDE_THRESHOLD_DEFAULT: f64 = 0.05;

/// Default for [`REVERT_BLEND_ALPHA_ENV`]. The overlay is off by
/// default so this Step 2 landing is a true no-op until the
/// production team picks a value through `predict_sweep`.
pub(crate) const REVERT_BLEND_ALPHA_DEFAULT: f64 = 0.0;

fn parse_revert_horizon_min_secs(value: Option<&str>) -> u64 {
    value
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(REVERT_HORIZON_MIN_SECS_DEFAULT)
}

fn parse_revert_magnitude_threshold(value: Option<&str>) -> f64 {
    value
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(REVERT_MAGNITUDE_THRESHOLD_DEFAULT)
}

fn parse_revert_blend_alpha(value: Option<&str>) -> f64 {
    value
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
        .unwrap_or(REVERT_BLEND_ALPHA_DEFAULT)
}

fn revert_horizon_min_secs() -> u64 {
    parse_revert_horizon_min_secs(std::env::var(REVERT_HORIZON_MIN_SECS_ENV).ok().as_deref())
}

fn revert_magnitude_threshold() -> f64 {
    parse_revert_magnitude_threshold(
        std::env::var(REVERT_MAGNITUDE_THRESHOLD_ENV)
            .ok()
            .as_deref(),
    )
}

fn revert_blend_alpha() -> f64 {
    parse_revert_blend_alpha(std::env::var(REVERT_BLEND_ALPHA_ENV).ok().as_deref())
}

/// Decide whether to apply the linear detrend pre-processor and return the
/// captured state when it is applied. See [`DETREND_SPAN_RATIO_THRESHOLD`]
/// for the rationale behind the slope-magnitude gate.
fn compute_detrend_state(
    characteristics: &TimeSeriesCharacteristics,
    log_values: &[f64],
    log_transformed: bool,
) -> Option<DetrendState> {
    if !adaptive_detrend_enabled() {
        return None;
    }
    if log_transformed || log_values.is_empty() {
        return None;
    }
    if characteristics.regime.regime == TimeSeriesRegime::MeanReverting {
        // VR test is used as a negative filter: never inject a trend into a
        // series that is statistically mean-reverting.
        return None;
    }
    let slope = characteristics.trend.slope;
    let intercept = characteristics.trend.intercept;
    if !slope.is_finite() || !intercept.is_finite() {
        return None;
    }
    let current_price = *log_values
        .last()
        .expect("log_values checked non-empty above");
    if !current_price.is_finite() || current_price.abs() < DETREND_PRICE_FLOOR {
        return None;
    }
    let span = slope.abs() * (log_values.len() - 1) as f64;
    let span_ratio = span / current_price.abs();
    if span_ratio <= DETREND_SPAN_RATIO_THRESHOLD {
        return None;
    }
    info!(
        slope = format!("{slope:.6}"),
        intercept = format!("{intercept:.6}"),
        span_ratio = format!("{span_ratio:.4}"),
        vr = format!("{:.4}", characteristics.regime.variance_ratio),
        regime = ?characteristics.regime.regime,
        "span_ratio above threshold, applying linear detrend"
    );
    Some(DetrendState {
        slope,
        intercept,
        training_len: log_values.len(),
    })
}

/// Add the linear trend back to a forecast vector. Forecast step i (0-indexed)
/// corresponds to original-domain index `training_len + i`.
///
/// The retrend adds two terms: the in-sample level at the last training
/// index (`slope × training_len + intercept`, constant across the
/// horizon) and a per-step extrapolation. The per-step extrapolation
/// uses the damped sum `slope × Σⱼ₌₁ⁱ φʲ` when `phi < 1.0`; when
/// `phi = 1.0` (the default), it collapses to the original undamped
/// `slope × i` and the formula reduces to `slope × (training_len + i)`.
fn add_trend_if_detrended(values: Vec<f64>, state: &Option<DetrendState>, phi: f64) -> Vec<f64> {
    match state {
        Some(s) => {
            let level_at_n = s.slope * s.training_len as f64 + s.intercept;
            let phi_eq_one = (phi - 1.0).abs() < 1e-12;
            values
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    let extra = if phi_eq_one {
                        s.slope * i as f64
                    } else {
                        // Σⱼ₌₁ⁱ φʲ = φ × (1 − φⁱ) / (1 − φ); for i = 0 this is 0.
                        s.slope * phi * (1.0 - phi.powi(i as i32)) / (1.0 - phi)
                    };
                    v + level_at_n + extra
                })
                .collect()
        }
        None => values,
    }
}

/// Clamp before exp() to prevent f64::INFINITY.
/// 709.0 = f64::MAX.ln().floor(); exp(709.78) ≈ f64::MAX, so 709.0 gives a safety margin.
/// NaN passes through so downstream f64s_to_decimals correctly rejects it.
fn safe_exp(v: f64) -> f64 {
    if v.is_nan() {
        return f64::NAN;
    }
    if v > 709.0 {
        tracing::warn!(value = v, "clamping exp() input to 709.0");
        709.0_f64.exp()
    } else {
        v.exp()
    }
}

/// Calculate median sampling interval in seconds.
fn calculate_median_interval(timestamps: &[NaiveDateTime]) -> i64 {
    if timestamps.len() < 2 {
        return 1;
    }

    let mut intervals: Vec<i64> = timestamps
        .array_windows()
        .map(|[a, b]| (*b - *a).num_seconds())
        .collect();
    let mid = intervals.len() / 2;
    (*intervals.select_nth_unstable(mid).1).max(1)
}

/// Convert TimeDelta to steps based on median sampling interval.
fn horizon_to_steps(horizon: &TimeDelta, timestamps: &[NaiveDateTime]) -> usize {
    if timestamps.len() < 2 {
        return 1;
    }

    let median_interval = calculate_median_interval(timestamps);

    // Convert horizon to steps (ceiling division, minimum 1)
    let horizon_secs = horizon.num_seconds();
    let steps = (horizon_secs.saturating_add(median_interval - 1) / median_interval) as usize;
    steps.max(1)
}

/// Convert TimeDelta to steps and generate forecast timestamps.
///
/// Returns (steps, forecast_timestamps) where:
/// - steps: number of forecast steps
/// - forecast_timestamps: timestamps for each forecast point
fn horizon_to_steps_with_timestamps(
    horizon: &TimeDelta,
    timestamps: &[NaiveDateTime],
) -> (usize, Vec<NaiveDateTime>) {
    let steps = horizon_to_steps(horizon, timestamps);

    if timestamps.is_empty() {
        return (steps, Vec::new());
    }

    let last_ts = *timestamps
        .last()
        .expect("timestamps verified non-empty above");
    let median_interval = calculate_median_interval(timestamps);

    let forecast_timestamps: Vec<NaiveDateTime> = (1..=steps)
        .map(|i| last_ts + TimeDelta::seconds(median_interval.saturating_mul(i as i64)))
        .collect();

    (steps, forecast_timestamps)
}

/// Calculate time budget based on data size.
fn calculate_time_budget(data_len: usize) -> f64 {
    // Base 60s, +10s per 100 points, max 900s
    let base = 60.0;
    let per_100_points = 10.0;
    (base + (data_len as f64 / 100.0) * per_100_points).min(900.0)
}

/// Main prediction entry point (backward-compatible free function).
///
/// Uses a default single-threaded pool. For controlled parallelism,
/// use [`Predictor::new`] with a specific thread count.
///
/// **Note**: The default `Predictor` is initialized once via `LazyLock`.
/// If initialization fails (e.g., thread pool creation error), the failure
/// is permanent for the lifetime of the process.
pub fn predict(input: &PredictionInput) -> Result<ForecastResult> {
    static DEFAULT: LazyLock<Result<Predictor>> = LazyLock::new(|| Predictor::new(1));
    let predictor = DEFAULT
        .as_ref()
        .map_err(|e| ChronosError::ModelError(format!("default predictor init failed: {e}")))?;
    predictor.predict(input)
}

mod revert_blend;

#[cfg(test)]
mod tests;
