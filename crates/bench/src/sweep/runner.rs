//! Single-job and full-sweep drivers.
//!
//! `run_one` evaluates the predictor on a single
//! `(series, history_len, horizon, eval_date)` combination and produces a
//! [`SweepRow`]. `run_sweep` (added in a later step) fans `run_one` across
//! the Cartesian product of a [`SweepConfig`].
//!
//! The forecast/actual alignment is grid-based: the predictor emits a
//! forecast on an equally-spaced timestamp grid derived from the training
//! interval, and the runner maps each grid point to its nearest actual in
//! the post-`eval_date` segment of the universe (within one median
//! sampling interval). Forecast steps with no matching actual truncate the
//! evaluated horizon; if no step can be aligned the job is rejected with
//! [`SweepError::InsufficientActual`].

use std::collections::BTreeMap;

use analyzer::TimeSeriesAnalyzer;
use chrono::{NaiveDateTime, TimeDelta};
use common::{BigDecimal, decimals_to_f64s};
use num_traits::ToPrimitive;
use predictor::{PredictionInput, Predictor};

use crate::sweep::safety::{
    try_compute_direction_metrics, try_compute_metrics, validate_current_value,
    validate_signal_threshold,
};
use crate::sweep::{SeriesInput, SweepError, SweepResult, SweepRow};

/// Minimum number of training samples required before a sweep job is
/// allowed to call the predictor. Below this the regime detector returns
/// its default `RandomWalk` label (analyzer requires 20 samples) and the
/// forecast accuracy is meaningless.
pub const MIN_TRAIN_SAMPLES: usize = 20;

/// Description of one sweep job. Carrying the full description by reference
/// keeps `run_one`'s signature small enough to satisfy
/// `clippy::too_many_arguments` without `#[allow]`.
#[derive(Debug, Clone, Copy)]
pub struct JobSpec<'a> {
    pub series: &'a SeriesInput,
    pub eval_date: NaiveDateTime,
    pub history_secs: i64,
    pub horizon_secs: i64,
    pub signal_threshold: f64,
}

/// Evaluate the predictor on a single job and return a fully-populated
/// `SweepRow`.
///
/// Errors are split into two camps: recoverable per-job failures
/// (`InsufficientHistory`, `InsufficientActual`, `DegenerateBaseline`)
/// which the caller will record on a row and continue with; and fatal
/// errors (`Chronos(ChronosError)`, `InvalidConfig`) which abort the run.
pub fn run_one(predictor: &Predictor, spec: JobSpec<'_>) -> SweepResult<SweepRow> {
    validate_signal_threshold(spec.signal_threshold)?;

    // ---- 1. Extract training subset and validate baseline. ----
    let train_subset = extract_train_subset(spec.series, spec.eval_date, spec.history_secs);
    if train_subset.len() < MIN_TRAIN_SAMPLES {
        return Err(SweepError::InsufficientHistory {
            eval_date: spec.eval_date,
            have: train_subset.len(),
            need: MIN_TRAIN_SAMPLES,
        });
    }

    let (train_last_ts, train_last_decimal) = train_subset
        .iter()
        .next_back()
        .expect("train_subset checked non-empty above");
    let train_last_ts = *train_last_ts;
    let train_last_value = train_last_decimal
        .to_f64()
        .ok_or(SweepError::DegenerateBaseline { value: f64::NAN })?;
    validate_current_value(train_last_value)?;

    let train_timestamps: Vec<NaiveDateTime> = train_subset.keys().copied().collect();
    let train_decimals: Vec<BigDecimal> = train_subset.values().cloned().collect();
    let train_values = decimals_to_f64s(&train_decimals)?;

    // ---- 2. Regime label for stratified aggregation. ----
    let analyzer = TimeSeriesAnalyzer::new();
    let regime_info = analyzer.detect_regime(&train_values);
    let regime_label = format!("{:?}", regime_info.regime);

    // ---- 3. Run the predictor. ----
    let horizon = TimeDelta::seconds(spec.horizon_secs);
    let prediction_input = PredictionInput {
        data: train_subset,
        horizon,
    };
    let result = predictor.predict(&prediction_input)?;

    let pred_timestamps: Vec<NaiveDateTime> = result.forecast_values.keys().copied().collect();
    let pred_decimals: Vec<BigDecimal> = result.forecast_values.values().cloned().collect();
    if pred_timestamps.is_empty() {
        return Err(SweepError::InsufficientActual {
            eval_date: spec.eval_date,
            have: 0,
            need: 1,
        });
    }
    let pred_values_full = decimals_to_f64s(&pred_decimals)?;

    // ---- 4. Align actuals to the forecast grid. ----
    let median_interval = median_sampling_interval(&train_timestamps);
    let align_tolerance_secs = median_interval.max(1);
    let alignment = align_actuals(
        &spec.series.data,
        spec.eval_date,
        &pred_timestamps,
        align_tolerance_secs,
    );
    if alignment.actual_values.is_empty() {
        return Err(SweepError::InsufficientActual {
            eval_date: spec.eval_date,
            have: 0,
            need: pred_timestamps.len(),
        });
    }

    let aligned_horizon = alignment.actual_values.len();
    let forecast_truncated = &pred_values_full[..aligned_horizon];

    // ---- 5. Metric computation. ----
    let direction = try_compute_direction_metrics(
        forecast_truncated,
        &alignment.actual_values,
        train_last_value,
        spec.signal_threshold,
    )?;
    let metrics = try_compute_metrics(
        forecast_truncated,
        &alignment.actual_values,
        &train_values,
        // Sweep does not carry per-series seasonal-period metadata, so we
        // evaluate MASE with the non-seasonal naïve baseline. Downstream
        // analysis can substitute a regime-aware baseline if needed.
        1,
    )?;

    // ---- 6. predicted_std endpoints. ----
    let (predicted_std_first, predicted_std_last) = match result.predicted_std.as_ref() {
        Some(map) => {
            let first = map.values().next().and_then(|d| d.to_f64());
            let last = map.values().next_back().and_then(|d| d.to_f64());
            (
                first.filter(|v| v.is_finite()),
                last.filter(|v| v.is_finite()),
            )
        }
        None => (None, None),
    };

    Ok(SweepRow {
        series_id: spec.series.series_id.clone(),
        eval_date: spec.eval_date,
        history_secs: spec.history_secs,
        horizon_secs: spec.horizon_secs,
        model_name: Some(result.model_name),
        strategy_name: Some(result.strategy_name),
        regime: Some(regime_label),

        train_last_ts: Some(train_last_ts),
        train_last_value: Some(train_last_value),
        pred_first_ts: Some(pred_timestamps[0]),
        actual_first_ts: alignment.first_actual_ts,
        aligned_horizon_steps: Some(aligned_horizon),
        max_alignment_gap_secs: Some(alignment.max_gap_secs),

        mae: finite(metrics.mae),
        rmse: finite(metrics.rmse),
        mape: finite(metrics.mape),
        mase: finite(metrics.mase),
        wape: finite(metrics.wape),

        dir_acc: finite(direction.dir_acc),
        dir_acc_filtered: direction.dir_acc_filtered.and_then(finite),
        filtered_count: Some(direction.filtered_count),
        per_row_ic: direction.ic.and_then(finite),
        calibration_residual: finite(direction.calibration_residual),

        predicted_std_first,
        predicted_std_last,

        processing_time_secs: finite(result.processing_time_secs),
        error: None,
        diagnostic_path: None,
    })
}

/// Slice the series so that the result contains every sample `ts` with
/// `eval_date - history_secs ≤ ts ≤ eval_date`.
fn extract_train_subset(
    series: &SeriesInput,
    eval_date: NaiveDateTime,
    history_secs: i64,
) -> BTreeMap<NaiveDateTime, BigDecimal> {
    let history_delta = TimeDelta::seconds(history_secs);
    let train_start = eval_date - history_delta;
    series
        .data
        .range(train_start..=eval_date)
        .map(|(ts, val)| (*ts, val.clone()))
        .collect()
}

/// Median of the consecutive-sample interval lengths, in seconds. Returns
/// 1 when the train segment has fewer than two timestamps.
fn median_sampling_interval(timestamps: &[NaiveDateTime]) -> i64 {
    if timestamps.len() < 2 {
        return 1;
    }
    let mut intervals: Vec<i64> = timestamps
        .array_windows()
        .map(|[a, b]| (*b - *a).num_seconds())
        .collect();
    intervals.sort_unstable();
    intervals[intervals.len() / 2]
}

/// Alignment result: a vector of actual values that line up with the
/// leading prefix of the forecast grid.
struct ActualAlignment {
    actual_values: Vec<f64>,
    first_actual_ts: Option<NaiveDateTime>,
    max_gap_secs: i64,
}

/// Build the `actual` vector by walking the forecast grid in order, and
/// for each grid step picking the universe sample with the closest
/// timestamp inside `tolerance_secs`. Universe samples are consumed left
/// to right, so each actual timestamp matches at most one forecast step.
///
/// The walk stops at the first forecast step that cannot be matched: the
/// returned vector is the longest prefix that could be aligned.
fn align_actuals(
    data: &BTreeMap<NaiveDateTime, BigDecimal>,
    eval_date: NaiveDateTime,
    forecast_grid: &[NaiveDateTime],
    tolerance_secs: i64,
) -> ActualAlignment {
    let mut actual_values = Vec::with_capacity(forecast_grid.len());
    let mut first_actual_ts = None;
    let mut max_gap_secs: i64 = 0;
    let mut cursor: Option<NaiveDateTime> = Some(eval_date);

    for &grid_ts in forecast_grid {
        let lower_bound = match cursor {
            Some(c) => c,
            None => eval_date,
        };
        let candidate = data
            .range(lower_bound..)
            .filter(|(ts, _)| **ts > eval_date)
            .min_by_key(|(ts, _)| (**ts - grid_ts).num_seconds().abs());
        let Some((ts, value)) = candidate else {
            break;
        };
        let gap_secs = (*ts - grid_ts).num_seconds().abs();
        if gap_secs > tolerance_secs {
            break;
        }
        let Some(v) = value.to_f64() else { break };
        if !v.is_finite() {
            break;
        }
        if first_actual_ts.is_none() {
            first_actual_ts = Some(*ts);
        }
        if gap_secs > max_gap_secs {
            max_gap_secs = gap_secs;
        }
        actual_values.push(v);
        cursor = ts.checked_add_signed(TimeDelta::seconds(1));
    }

    ActualAlignment {
        actual_values,
        first_actual_ts,
        max_gap_secs,
    }
}

/// Convert a metric value to `Option<f64>`, mapping NaN / ±Inf to `None`.
fn finite(value: f64) -> Option<f64> {
    if value.is_finite() { Some(value) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, TimeDelta};
    use num_traits::FromPrimitive;

    fn ts(hour_offset: i64) -> NaiveDateTime {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        base + TimeDelta::hours(hour_offset)
    }

    fn series_with_linear_trend(n: usize) -> SeriesInput {
        let data: BTreeMap<NaiveDateTime, BigDecimal> = (0..n)
            .map(|i| {
                let v = 100.0 + i as f64;
                (ts(i as i64), BigDecimal::from_f64(v).unwrap())
            })
            .collect();
        SeriesInput {
            series_id: "synthetic".into(),
            data,
        }
    }

    #[test]
    fn extract_train_subset_respects_window() {
        let series = series_with_linear_trend(100);
        let eval_date = ts(60);
        let history_secs = TimeDelta::hours(20).num_seconds();
        let train = extract_train_subset(&series, eval_date, history_secs);
        // window is [eval_date - 20h, eval_date] => 21 hourly samples.
        assert_eq!(train.len(), 21);
        assert_eq!(*train.keys().next().unwrap(), ts(40));
        assert_eq!(*train.keys().next_back().unwrap(), ts(60));
    }

    #[test]
    fn median_sampling_interval_handles_irregular_data() {
        let timestamps = vec![ts(0), ts(1), ts(2), ts(4), ts(7)];
        // intervals: 3600, 3600, 7200, 10800 -> sorted -> median at idx 2 = 7200
        assert_eq!(median_sampling_interval(&timestamps), 7200);
    }

    #[test]
    fn align_actuals_truncates_when_universe_runs_out() {
        let series = series_with_linear_trend(50);
        let eval_date = ts(40);
        let grid = (1..=20).map(|i| ts(40 + i)).collect::<Vec<_>>();
        let tolerance = TimeDelta::hours(1).num_seconds();
        let alignment = align_actuals(&series.data, eval_date, &grid, tolerance);
        // universe has data up to ts(49), so alignment covers grid 1..=9.
        assert_eq!(alignment.actual_values.len(), 9);
        assert_eq!(alignment.first_actual_ts, Some(ts(41)));
        assert!(alignment.max_gap_secs <= tolerance);
    }

    #[test]
    fn align_actuals_returns_empty_when_no_data_after_eval_date() {
        let series = series_with_linear_trend(20);
        let eval_date = ts(20);
        let grid = vec![ts(21), ts(22)];
        let alignment = align_actuals(&series.data, eval_date, &grid, 3600);
        assert!(alignment.actual_values.is_empty());
        assert_eq!(alignment.first_actual_ts, None);
    }

    #[test]
    fn finite_helper_filters_non_finite() {
        assert_eq!(finite(1.5), Some(1.5));
        assert_eq!(finite(f64::NAN), None);
        assert_eq!(finite(f64::INFINITY), None);
        assert_eq!(finite(f64::NEG_INFINITY), None);
    }

    #[test]
    fn run_one_returns_insufficient_history_for_short_window() {
        let series = series_with_linear_trend(200);
        let predictor = Predictor::new(1).expect("predictor init");
        let spec = JobSpec {
            series: &series,
            eval_date: ts(10),
            history_secs: TimeDelta::hours(5).num_seconds(),
            horizon_secs: TimeDelta::hours(4).num_seconds(),
            signal_threshold: crate::sweep::DEFAULT_SIGNAL_THRESHOLD,
        };
        match run_one(&predictor, spec) {
            Err(SweepError::InsufficientHistory { have, need, .. }) => {
                assert!(have < need);
                assert_eq!(need, MIN_TRAIN_SAMPLES);
            }
            other => panic!("expected InsufficientHistory, got {other:?}"),
        }
    }

    #[test]
    fn run_one_returns_insufficient_actual_when_eval_date_is_at_universe_end() {
        let series = series_with_linear_trend(100);
        let predictor = Predictor::new(1).expect("predictor init");
        let spec = JobSpec {
            series: &series,
            // eval_date past the last sample → no actuals available.
            eval_date: ts(99),
            history_secs: TimeDelta::hours(50).num_seconds(),
            horizon_secs: TimeDelta::hours(10).num_seconds(),
            signal_threshold: crate::sweep::DEFAULT_SIGNAL_THRESHOLD,
        };
        match run_one(&predictor, spec) {
            Err(SweepError::InsufficientActual { .. }) => {}
            other => panic!("expected InsufficientActual, got {other:?}"),
        }
    }

    #[test]
    fn run_one_produces_populated_row_for_linear_universe() {
        let series = series_with_linear_trend(150);
        let predictor = Predictor::new(1).expect("predictor init");
        let spec = JobSpec {
            series: &series,
            eval_date: ts(100),
            history_secs: TimeDelta::hours(60).num_seconds(),
            horizon_secs: TimeDelta::hours(20).num_seconds(),
            signal_threshold: crate::sweep::DEFAULT_SIGNAL_THRESHOLD,
        };
        let row = run_one(&predictor, spec).expect("run_one should succeed");
        assert_eq!(row.series_id, "synthetic");
        assert_eq!(row.eval_date, ts(100));
        assert!(row.regime.is_some());
        assert_eq!(row.train_last_ts, Some(ts(100)));
        assert!(row.train_last_value.is_some());
        assert!(row.aligned_horizon_steps.unwrap() > 0);
        assert!(row.mae.is_some());
        assert!(row.dir_acc.is_some());
        assert!(row.error.is_none());
    }
}
