//! `run_sweep` must produce byte-identical output regardless of the
//! number of worker threads. This test runs the same sweep twice — once
//! with a single worker, once with four — and asserts that every metric
//! field of every row is equal.
//!
//! The determinism guarantee rests on three sweep-internal choices:
//! - each worker owns an independent `Predictor::new(1)`, so the inner
//!   rayon `par_iter` inside `train_hierarchically` cannot cross-steal
//!   between sweep jobs;
//! - the result vector is sorted by `(series_id, eval_date,
//!   history_secs, horizon_secs)` after the join;
//! - all per-job error paths are deterministic (no clock-based timeouts
//!   in the metric helpers).

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use bench::sweep::{
    DEFAULT_SIGNAL_THRESHOLD, SeriesInput, SweepConfig, SweepRow, default_calibration_buckets,
    run_sweep,
};
use bigdecimal::BigDecimal;
use chrono::{NaiveDate, NaiveDateTime, TimeDelta};
use num_traits::FromPrimitive;

fn ts(hour_offset: i64) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        + TimeDelta::hours(hour_offset)
}

fn linear_series(id: &str, slope: f64, n: usize) -> SeriesInput {
    let data: BTreeMap<NaiveDateTime, BigDecimal> = (0..n)
        .map(|i| {
            let v = 100.0 + i as f64 * slope;
            (ts(i as i64), BigDecimal::from_f64(v).unwrap())
        })
        .collect();
    SeriesInput {
        series_id: id.to_string(),
        data,
    }
}

fn build_config(workers: usize) -> SweepConfig {
    SweepConfig {
        series_universe: vec![
            linear_series("alpha", 1.0, 200),
            linear_series("bravo", 0.5, 200),
            linear_series("charlie", -0.25, 200),
        ],
        history_lens_secs: vec![
            TimeDelta::hours(60).num_seconds(),
            TimeDelta::hours(90).num_seconds(),
        ],
        horizons_secs: vec![TimeDelta::hours(12).num_seconds()],
        eval_dates: vec![ts(120), ts(150)],
        signal_threshold: DEFAULT_SIGNAL_THRESHOLD,
        calibration_buckets: default_calibration_buckets(),
        workers: NonZeroUsize::new(workers).unwrap(),
        diagnostic_dir: None,
    }
}

fn assert_rows_equal(a: &SweepRow, b: &SweepRow, idx: usize) {
    assert_eq!(a.series_id, b.series_id, "row {idx}: series_id mismatch");
    assert_eq!(a.eval_date, b.eval_date, "row {idx}: eval_date mismatch");
    assert_eq!(a.history_secs, b.history_secs, "row {idx}: history_secs");
    assert_eq!(a.horizon_secs, b.horizon_secs, "row {idx}: horizon_secs");
    assert_eq!(a.regime, b.regime, "row {idx}: regime");
    assert_eq!(a.train_last_ts, b.train_last_ts, "row {idx}: train_last_ts");
    assert_eq!(
        a.aligned_horizon_steps, b.aligned_horizon_steps,
        "row {idx}: aligned_horizon_steps"
    );
    // Metric comparisons go through bit-wise equality so a NaN escape
    // would be caught even if assert_eq's PartialEq round-trips it.
    fn opt_bits(x: Option<f64>) -> Option<u64> {
        x.map(f64::to_bits)
    }
    assert_eq!(opt_bits(a.mae), opt_bits(b.mae), "row {idx}: mae");
    assert_eq!(opt_bits(a.rmse), opt_bits(b.rmse), "row {idx}: rmse");
    assert_eq!(opt_bits(a.mape), opt_bits(b.mape), "row {idx}: mape");
    assert_eq!(opt_bits(a.mase), opt_bits(b.mase), "row {idx}: mase");
    assert_eq!(opt_bits(a.wape), opt_bits(b.wape), "row {idx}: wape");
    assert_eq!(
        opt_bits(a.dir_acc),
        opt_bits(b.dir_acc),
        "row {idx}: dir_acc"
    );
    assert_eq!(
        opt_bits(a.dir_acc_filtered),
        opt_bits(b.dir_acc_filtered),
        "row {idx}: dir_acc_filtered"
    );
    assert_eq!(
        opt_bits(a.per_row_ic),
        opt_bits(b.per_row_ic),
        "row {idx}: per_row_ic"
    );
    assert_eq!(
        opt_bits(a.calibration_residual),
        opt_bits(b.calibration_residual),
        "row {idx}: calibration_residual"
    );
    assert_eq!(
        opt_bits(a.final_pred_return),
        opt_bits(b.final_pred_return),
        "row {idx}: final_pred_return"
    );
    assert_eq!(
        opt_bits(a.final_actual_return),
        opt_bits(b.final_actual_return),
        "row {idx}: final_actual_return"
    );
    assert_eq!(a.error, b.error, "row {idx}: error");
}

#[test]
fn run_sweep_is_deterministic_across_worker_counts() {
    let single = run_sweep(build_config(1)).expect("workers=1 sweep");
    let parallel = run_sweep(build_config(4)).expect("workers=4 sweep");
    assert_eq!(
        single.rows.len(),
        parallel.rows.len(),
        "row count mismatch: single={}, parallel={}",
        single.rows.len(),
        parallel.rows.len()
    );
    for (i, (a, b)) in single.rows.iter().zip(parallel.rows.iter()).enumerate() {
        assert_rows_equal(a, b, i);
    }
    // Cross-sectional IC depends on the same row vector, so it must also
    // round-trip identically.
    assert_eq!(
        single
            .cross_sectional_ic_by_date
            .iter()
            .map(|(k, v)| (*k, v.map(f64::to_bits)))
            .collect::<Vec<_>>(),
        parallel
            .cross_sectional_ic_by_date
            .iter()
            .map(|(k, v)| (*k, v.map(f64::to_bits)))
            .collect::<Vec<_>>(),
    );
}
