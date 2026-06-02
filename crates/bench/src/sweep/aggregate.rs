//! Aggregations applied to a finished sweep before the report is emitted.
//!
//! All aggregations skip rows where the underlying metric is `None`
//! (whether the row was skipped or the metric itself was NaN/Inf), so the
//! averages reported below are strictly over the finite, computed subset.
//! The denominator (`n`) makes that subset visible to consumers.
//!
//! Cross-sectional Information Coefficient is computed across rows whose
//! `eval_date` matches and which have both `final_pred_return` and
//! `final_actual_return` finite. We reuse the Spearman implementation
//! from `crate::direction_metrics::spearman_correlation` so the
//! per-row IC and the cross-sectional IC agree on the tie-handling rule.

use std::collections::BTreeMap;

use chrono::NaiveDateTime;

use crate::direction_metrics::spearman_correlation;
use crate::sweep::{CROSS_SECTIONAL_MIN_N, RegimeStats, SeriesStats, SweepRow};

/// Bucket rows by regime label and compute the per-bucket averages
/// reported on `SweepReport.regime_summary`. Rows without a regime label
/// (i.e. skipped before regime detection) are ignored. Output is sorted
/// by regime label for stable serialisation.
pub fn aggregate_by_regime(rows: &[SweepRow]) -> Vec<RegimeStats> {
    let mut by_regime: BTreeMap<String, Vec<&SweepRow>> = BTreeMap::new();
    for row in rows {
        if let Some(regime) = row.regime.as_ref() {
            by_regime.entry(regime.clone()).or_default().push(row);
        }
    }
    by_regime
        .into_iter()
        .map(|(regime, group)| RegimeStats {
            n: group.len(),
            avg_dir_acc: avg_finite(group.iter().filter_map(|r| r.dir_acc)),
            avg_dir_acc_filtered: avg_finite(group.iter().filter_map(|r| r.dir_acc_filtered)),
            avg_per_row_ic: avg_finite(group.iter().filter_map(|r| r.per_row_ic)),
            avg_calibration_residual: avg_finite(
                group.iter().filter_map(|r| r.calibration_residual),
            ),
            avg_mape: avg_finite(group.iter().filter_map(|r| r.mape)),
            avg_mase: avg_finite(group.iter().filter_map(|r| r.mase)),
            regime,
        })
        .collect()
}

/// Bucket rows by `series_id` and compute the per-series averages reported
/// on `SweepReport.series_summary`. Skipped rows still count in `n` so the
/// consumer can see attempted-vs-successful job counts.
pub fn aggregate_by_series(rows: &[SweepRow]) -> Vec<SeriesStats> {
    let mut by_series: BTreeMap<String, Vec<&SweepRow>> = BTreeMap::new();
    for row in rows {
        by_series
            .entry(row.series_id.clone())
            .or_default()
            .push(row);
    }
    by_series
        .into_iter()
        .map(|(series_id, group)| SeriesStats {
            n: group.len(),
            avg_dir_acc: avg_finite(group.iter().filter_map(|r| r.dir_acc)),
            avg_per_row_ic: avg_finite(group.iter().filter_map(|r| r.per_row_ic)),
            series_id,
        })
        .collect()
}

/// Compute Spearman rank correlation between `final_pred_return` and
/// `final_actual_return` for every distinct `eval_date`. Returns two maps
/// keyed by date:
/// - The IC value (`None` when fewer than [`CROSS_SECTIONAL_MIN_N`] series
///   contributed or when one of the input series has zero variance).
/// - The number of contributing series (always reported so the caller
///   can re-pick the threshold downstream).
pub fn cross_sectional_ic(
    rows: &[SweepRow],
) -> (
    BTreeMap<NaiveDateTime, Option<f64>>,
    BTreeMap<NaiveDateTime, usize>,
) {
    let mut by_date: BTreeMap<NaiveDateTime, Vec<(f64, f64)>> = BTreeMap::new();
    for row in rows {
        if let (Some(p), Some(a)) = (row.final_pred_return, row.final_actual_return)
            && p.is_finite()
            && a.is_finite()
        {
            by_date.entry(row.eval_date).or_default().push((p, a));
        }
    }

    let mut ic_by_date = BTreeMap::new();
    let mut n_by_date = BTreeMap::new();
    for (date, pairs) in by_date {
        let n = pairs.len();
        n_by_date.insert(date, n);
        if n < CROSS_SECTIONAL_MIN_N {
            ic_by_date.insert(date, None);
            continue;
        }
        let preds: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
        let actuals: Vec<f64> = pairs.iter().map(|(_, a)| *a).collect();
        ic_by_date.insert(date, spearman_correlation(&preds, &actuals));
    }

    (ic_by_date, n_by_date)
}

fn avg_finite<I: IntoIterator<Item = f64>>(values: I) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in values {
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    if n == 0 { None } else { Some(sum / n as f64) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn ts(hour: i64) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            + chrono::TimeDelta::hours(hour)
    }

    fn base_row(series_id: &str, eval_date: NaiveDateTime, regime: &str) -> SweepRow {
        SweepRow {
            series_id: series_id.into(),
            eval_date,
            history_secs: 3600,
            horizon_secs: 600,
            model_name: Some("test".into()),
            strategy_name: Some("test".into()),
            regime: Some(regime.into()),
            train_last_ts: Some(eval_date),
            train_last_value: Some(1.0),
            pred_first_ts: Some(eval_date),
            actual_first_ts: Some(eval_date),
            aligned_horizon_steps: Some(1),
            max_alignment_gap_secs: Some(0),
            mae: Some(0.1),
            rmse: Some(0.1),
            mape: Some(1.0),
            mase: Some(0.5),
            wape: Some(0.5),
            dir_acc: Some(0.6),
            dir_acc_filtered: Some(0.7),
            filtered_count: Some(5),
            per_row_ic: Some(0.2),
            calibration_residual: Some(0.0),
            predicted_std_first: None,
            predicted_std_last: None,
            final_pred_return: None,
            final_actual_return: None,
            processing_time_secs: None,
            error: None,
            diagnostic_path: None,
        }
    }

    #[test]
    fn aggregate_by_regime_groups_and_averages() {
        let mut a = base_row("S1", ts(0), "Trending");
        a.dir_acc = Some(0.8);
        let mut b = base_row("S2", ts(0), "Trending");
        b.dir_acc = Some(0.6);
        let mut c = base_row("S3", ts(0), "RandomWalk");
        c.dir_acc = Some(0.5);
        let agg = aggregate_by_regime(&[a, b, c]);
        assert_eq!(agg.len(), 2);
        let trending = agg.iter().find(|r| r.regime == "Trending").unwrap();
        assert_eq!(trending.n, 2);
        assert!((trending.avg_dir_acc.unwrap() - 0.7).abs() < 1e-12);
        let rw = agg.iter().find(|r| r.regime == "RandomWalk").unwrap();
        assert_eq!(rw.n, 1);
        assert_eq!(rw.avg_dir_acc, Some(0.5));
    }

    #[test]
    fn aggregate_by_regime_skips_rows_without_label() {
        let mut a = base_row("S1", ts(0), "Trending");
        a.regime = None;
        let agg = aggregate_by_regime(&[a]);
        assert!(agg.is_empty());
    }

    #[test]
    fn aggregate_by_series_counts_attempted_rows() {
        let mut a = base_row("BTC", ts(0), "Trending");
        let b = base_row("ETH", ts(0), "Trending");
        a.error = Some("skipped".into()); // skipped row still counts in n.
        a.dir_acc = None;
        let agg = aggregate_by_series(&[a, b]);
        let btc = agg.iter().find(|r| r.series_id == "BTC").unwrap();
        assert_eq!(btc.n, 1);
        assert!(btc.avg_dir_acc.is_none());
    }

    #[test]
    fn cross_sectional_ic_requires_min_series() {
        // With fewer series than CROSS_SECTIONAL_MIN_N (=10) the IC is None
        // even when the inputs are perfectly correlated.
        let mut rows = Vec::new();
        for i in 0..5 {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let (ic, n) = cross_sectional_ic(&rows);
        assert_eq!(n[&ts(0)], 5);
        assert_eq!(ic[&ts(0)], None);
    }

    #[test]
    fn cross_sectional_ic_computes_above_threshold() {
        let mut rows = Vec::new();
        for i in 0..CROSS_SECTIONAL_MIN_N {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let (ic, n) = cross_sectional_ic(&rows);
        assert_eq!(n[&ts(0)], CROSS_SECTIONAL_MIN_N);
        let ic_value = ic[&ts(0)].expect("ic computable above threshold");
        assert!((ic_value - 1.0).abs() < 1e-12, "got {ic_value}");
    }

    #[test]
    fn cross_sectional_ic_skips_rows_with_missing_returns() {
        let mut rows = Vec::new();
        for i in 0..CROSS_SECTIONAL_MIN_N {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.final_pred_return = Some(i as f64);
            // First row has no actual return → excluded.
            if i != 0 {
                r.final_actual_return = Some(i as f64);
            }
            rows.push(r);
        }
        let (_, n) = cross_sectional_ic(&rows);
        assert_eq!(n[&ts(0)], CROSS_SECTIONAL_MIN_N - 1);
    }

    #[test]
    fn cross_sectional_ic_partitions_by_eval_date() {
        let mut rows = Vec::new();
        for i in 0..CROSS_SECTIONAL_MIN_N {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let mut r = base_row("X", ts(24), "Trending");
        r.final_pred_return = Some(0.0);
        r.final_actual_return = Some(0.0);
        rows.push(r);
        let (_, n) = cross_sectional_ic(&rows);
        assert_eq!(n[&ts(0)], CROSS_SECTIONAL_MIN_N);
        assert_eq!(n[&ts(24)], 1);
    }
}
