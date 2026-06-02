use std::num::NonZeroUsize;

use chrono::NaiveDate;

use super::*;

fn sample_ts() -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 6, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

#[test]
fn test_default_calibration_buckets_cover_outer_ranges() {
    let buckets = default_calibration_buckets();
    assert_eq!(buckets.first().unwrap().0, -10.0);
    assert_eq!(buckets.last().unwrap().1, 10.0);
    // Buckets must form a non-overlapping, contiguous sequence.
    for window in buckets.windows(2) {
        assert!(
            (window[0].1 - window[1].0).abs() < 1e-12,
            "bucket boundary mismatch: {:?} → {:?}",
            window[0],
            window[1],
        );
    }
}

#[test]
fn test_sweep_row_skipped_carries_reason() {
    let row = SweepRow::skipped(
        "BTC".to_string(),
        sample_ts(),
        86_400 * 30,
        3_600 * 24,
        "no data".to_string(),
    );
    assert_eq!(row.series_id, "BTC");
    assert_eq!(row.history_secs, 86_400 * 30);
    assert_eq!(row.horizon_secs, 3_600 * 24);
    assert_eq!(row.error.as_deref(), Some("no data"));
    assert!(row.mae.is_none());
    assert!(row.dir_acc.is_none());
    assert!(row.regime.is_none());
}

#[test]
fn test_sweep_config_constructs_with_required_fields() {
    let config = SweepConfig {
        series_universe: vec![],
        history_lens_secs: vec![86_400 * 7],
        horizons_secs: vec![3_600 * 24],
        eval_dates: vec![sample_ts()],
        signal_threshold: DEFAULT_SIGNAL_THRESHOLD,
        calibration_buckets: default_calibration_buckets(),
        workers: NonZeroUsize::new(2).unwrap(),
        diagnostic_dir: None,
    };
    assert_eq!(config.workers.get(), 2);
    assert_eq!(config.signal_threshold, DEFAULT_SIGNAL_THRESHOLD);
}

#[test]
fn test_sweep_error_display() {
    let err = SweepError::InsufficientHistory {
        eval_date: sample_ts(),
        have: 3,
        need: 30,
    };
    let msg = format!("{err}");
    assert!(msg.contains("insufficient training history"));
    assert!(msg.contains("have 3"));
    assert!(msg.contains("need at least 30"));
}
