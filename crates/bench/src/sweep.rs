//! Cross-product evaluation of the prediction pipeline.
//!
//! `run_sweep` evaluates the predictor over a Cartesian product of
//! `(series × history_len × horizon × eval_date)` and aggregates the
//! direction-based metrics from [`crate::direction_metrics`] alongside the
//! magnitude metrics from [`crate::metrics`]. It is the chronos-rs analog of
//! the `predict_sweep` binary in the zaciraci workspace.
//!
//! The CLI driver lives in `bin/predict_sweep.rs` and is gated behind the
//! `cli` feature. The library types in this module have no CLI dependencies.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use chrono::NaiveDateTime;
use common::{BigDecimal, ChronosError};
use serde::{Deserialize, Serialize};

/// Default magnitude threshold used to filter out flat predictions when
/// computing `dir_acc_filtered`. Matches the threshold used in zaciraci's
/// predict_sweep evaluation (0.5% return).
pub const DEFAULT_SIGNAL_THRESHOLD: f64 = 0.005;

/// Calibration bucket boundaries used by default. The outer buckets cover
/// `[-1000%, -5%)` and `[+5%, +1000%)` to avoid silently dropping high-volatility
/// meme-coin returns that exceed the conventional ±100% range.
pub fn default_calibration_buckets() -> Vec<(f64, f64)> {
    vec![
        (-10.0, -0.05),
        (-0.05, -0.01),
        (-0.01, -0.005),
        (-0.005, 0.005),
        (0.005, 0.01),
        (0.01, 0.05),
        (0.05, 10.0),
    ]
}

/// One time-series in the sweep universe. `data` must be non-empty and
/// sorted by timestamp (guaranteed by `BTreeMap`).
#[derive(Debug, Clone)]
pub struct SeriesInput {
    pub series_id: String,
    pub data: BTreeMap<NaiveDateTime, BigDecimal>,
}

/// Sweep configuration.
///
/// The Cartesian product
/// `series_universe × history_lens_secs × horizons_secs × eval_dates`
/// defines the set of jobs evaluated by `run_sweep`.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub series_universe: Vec<SeriesInput>,
    pub history_lens_secs: Vec<i64>,
    pub horizons_secs: Vec<i64>,
    pub eval_dates: Vec<NaiveDateTime>,
    pub signal_threshold: f64,
    pub calibration_buckets: Vec<(f64, f64)>,
    pub workers: NonZeroUsize,
    /// When set, raw forecast / actual vectors for each job are dumped as
    /// JSON files in this directory for offline diagnosis. The path written
    /// is recorded in `SweepRow::diagnostic_path`.
    pub diagnostic_dir: Option<PathBuf>,
}

/// One row of the sweep output, one per `(series, history, horizon, eval_date)`
/// job. Metric fields use `Option<f64>` so that NaN / Inf / "skipped" are
/// expressible distinctly from a true zero — see
/// `crate::sweep::output::serialize_finite_opt`.
///
/// When a job fails the row is still emitted with `error` set and metric
/// fields left as `None`; the I/O-contract metadata captures the state of
/// the train/actual subset that was reached before the failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepRow {
    // --- identification ---
    pub series_id: String,
    pub eval_date: NaiveDateTime,
    pub history_secs: i64,
    pub horizon_secs: i64,
    pub model_name: Option<String>,
    pub strategy_name: Option<String>,
    pub regime: Option<String>,

    // --- I/O contract metadata ---
    /// Last training timestamp actually used (may be < `eval_date` if the
    /// universe has gaps).
    pub train_last_ts: Option<NaiveDateTime>,
    /// Baseline value at `train_last_ts` in the original (un-normalized)
    /// scale. Used as the `current_value` for return calculations.
    pub train_last_value: Option<f64>,
    /// First forecast timestamp emitted by the predictor.
    pub pred_first_ts: Option<NaiveDateTime>,
    /// First actual timestamp matched to a forecast step on the eval grid.
    pub actual_first_ts: Option<NaiveDateTime>,
    /// Number of forecast steps with a matched actual value.
    pub aligned_horizon_steps: Option<usize>,
    /// Maximum gap (seconds) between a forecast-grid step and the nearest
    /// actual timestamp used for alignment. Large gaps indicate sparse data.
    pub max_alignment_gap_secs: Option<i64>,

    // --- magnitude metrics ---
    pub mae: Option<f64>,
    pub rmse: Option<f64>,
    pub mape: Option<f64>,
    pub mase: Option<f64>,
    pub wape: Option<f64>,

    // --- direction metrics ---
    pub dir_acc: Option<f64>,
    pub dir_acc_filtered: Option<f64>,
    pub filtered_count: Option<usize>,
    /// Per-row time-series IC (Spearman correlation of pred/actual return
    /// series within a single sweep row). Cross-sectional IC is computed
    /// across rows on `SweepReport`.
    pub per_row_ic: Option<f64>,
    pub calibration_residual: Option<f64>,

    // --- uncertainty ---
    pub predicted_std_first: Option<f64>,
    pub predicted_std_last: Option<f64>,

    // --- bookkeeping ---
    pub processing_time_secs: Option<f64>,
    pub error: Option<String>,
    pub diagnostic_path: Option<PathBuf>,
}

impl SweepRow {
    /// Build a minimal row for a job that was skipped before any prediction
    /// could be attempted (e.g. insufficient history).
    pub fn skipped(
        series_id: String,
        eval_date: NaiveDateTime,
        history_secs: i64,
        horizon_secs: i64,
        reason: String,
    ) -> Self {
        Self {
            series_id,
            eval_date,
            history_secs,
            horizon_secs,
            model_name: None,
            strategy_name: None,
            regime: None,
            train_last_ts: None,
            train_last_value: None,
            pred_first_ts: None,
            actual_first_ts: None,
            aligned_horizon_steps: None,
            max_alignment_gap_secs: None,
            mae: None,
            rmse: None,
            mape: None,
            mase: None,
            wape: None,
            dir_acc: None,
            dir_acc_filtered: None,
            filtered_count: None,
            per_row_ic: None,
            calibration_residual: None,
            predicted_std_first: None,
            predicted_std_last: None,
            processing_time_secs: None,
            error: Some(reason),
            diagnostic_path: None,
        }
    }
}

/// Aggregate stats per regime label, computed by `aggregate_by_regime`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeStats {
    pub regime: String,
    pub n: usize,
    pub avg_dir_acc: Option<f64>,
    pub avg_dir_acc_filtered: Option<f64>,
    pub avg_per_row_ic: Option<f64>,
    pub avg_calibration_residual: Option<f64>,
    pub avg_mape: Option<f64>,
    pub avg_mase: Option<f64>,
}

/// Aggregate stats per series, computed by `aggregate_by_series`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStats {
    pub series_id: String,
    pub n: usize,
    pub avg_dir_acc: Option<f64>,
    pub avg_per_row_ic: Option<f64>,
}

/// Result of a sweep run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    pub rows: Vec<SweepRow>,
    pub regime_summary: Vec<RegimeStats>,
    pub series_summary: Vec<SeriesStats>,
    /// Cross-sectional Spearman correlation of predicted vs actual return
    /// across all series for a single `eval_date`. Keyed by `eval_date`.
    /// `None` when fewer than `cross_sectional_min_n` series contributed or
    /// when variance is degenerate.
    pub cross_sectional_ic_by_date: BTreeMap<NaiveDateTime, Option<f64>>,
    /// Number of series that contributed to each cross-sectional IC.
    pub cross_sectional_ic_n: BTreeMap<NaiveDateTime, usize>,
}

/// Minimum number of series required for a cross-sectional IC to be
/// statistically meaningful. Below this threshold the value is reported
/// as `None` even when the rank correlation is computable.
pub const CROSS_SECTIONAL_MIN_N: usize = 10;

/// Errors raised by the sweep driver.
///
/// `InsufficientHistory` and `InsufficientActual` are recoverable per-job
/// errors — the driver records the row with `error = Some(...)` and
/// continues. Other variants are fatal and stop the run.
#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    #[error("chronos: {0}")]
    Chronos(#[from] ChronosError),
    #[error(
        "insufficient training history at {eval_date}: have {have} samples, need at least {need}"
    )]
    InsufficientHistory {
        eval_date: NaiveDateTime,
        have: usize,
        need: usize,
    },
    #[error(
        "insufficient actual data after {eval_date}: have {have} samples, need at least {need}"
    )]
    InsufficientActual {
        eval_date: NaiveDateTime,
        have: usize,
        need: usize,
    },
    #[error("baseline value {value} is degenerate (zero, non-finite, or near-underflow)")]
    DegenerateBaseline { value: f64 },
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("worker panicked while running job: {message}")]
    WorkerPanic { message: String },
}

/// Convenience alias for results returned by sweep functions.
pub type SweepResult<T> = Result<T, SweepError>;

pub mod runner;
pub mod safety;

pub use runner::{JobSpec, MIN_TRAIN_SAMPLES, run_one};

#[cfg(test)]
mod tests;
