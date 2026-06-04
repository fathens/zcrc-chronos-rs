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
use crate::sweep::{
    CROSS_SECTIONAL_MIN_N, FLAT_RETURN_EPSILON, HorizonStats, RegimeStats, SeriesStats, SweepRow,
};

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

/// Bucket rows by `horizon_secs` and compute per-horizon diagnostics
/// (flat rate, average IC, cross-sectional IC across all rows at that
/// horizon, decile spread). Output is sorted by `horizon_secs` for stable
/// serialisation.
///
/// The cross-sectional IC pools every row at the same horizon — across
/// `eval_date`, `history_secs`, and `series_id` — so the result reflects
/// directional skill at that horizon regardless of timing. Decile spread
/// pools the same set.
pub fn aggregate_by_horizon(rows: &[SweepRow]) -> Vec<HorizonStats> {
    let mut by_horizon: BTreeMap<i64, Vec<&SweepRow>> = BTreeMap::new();
    for row in rows {
        by_horizon.entry(row.horizon_secs).or_default().push(row);
    }
    by_horizon
        .into_iter()
        .map(|(horizon_secs, group)| {
            let n = group.len();
            let flat_count = group
                .iter()
                .filter(|r| {
                    r.final_pred_return
                        .is_some_and(|p| p.is_finite() && p.abs() < FLAT_RETURN_EPSILON)
                })
                .count();

            let pairs: Vec<(f64, f64)> = group
                .iter()
                .filter_map(|r| match (r.final_pred_return, r.final_actual_return) {
                    (Some(p), Some(a)) if p.is_finite() && a.is_finite() => Some((p, a)),
                    _ => None,
                })
                .collect();

            let cross_sectional_n = pairs.len();
            let cross_sectional_ic = if cross_sectional_n < CROSS_SECTIONAL_MIN_N {
                None
            } else {
                let preds: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
                let actuals: Vec<f64> = pairs.iter().map(|(_, a)| *a).collect();
                spearman_correlation(&preds, &actuals)
            };

            HorizonStats {
                horizon_secs,
                n,
                flat_count,
                avg_dir_acc: avg_finite(group.iter().filter_map(|r| r.dir_acc)),
                avg_per_row_ic: avg_finite(group.iter().filter_map(|r| r.per_row_ic)),
                cross_sectional_ic,
                cross_sectional_n,
                decile_spread: decile_spread(&pairs),
            }
        })
        .collect()
}

/// Mean `actual` of the top predicted decile minus mean of the bottom
/// decile. Pairs are sorted by `pred`; the highest 10 % and the lowest
/// 10 % each provide at least one observation only when the input has at
/// least ten finite-return rows. Returns `None` otherwise.
fn decile_spread(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 10 {
        return None;
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // chunk size: ceil(n / 10) so the partition always produces a top and
    // bottom slice with at least 1 element and at most ceil(n/10) each.
    let chunk = sorted.len().div_ceil(10);
    let bottom_mean = mean_of_actuals(&sorted[..chunk])?;
    let top_mean = mean_of_actuals(&sorted[sorted.len() - chunk..])?;
    Some(top_mean - bottom_mean)
}

fn mean_of_actuals(slice: &[(f64, f64)]) -> Option<f64> {
    if slice.is_empty() {
        return None;
    }
    let n = slice.len() as f64;
    Some(slice.iter().map(|(_, a)| *a).sum::<f64>() / n)
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
    fn aggregate_by_horizon_counts_flat_predictions() {
        let mut rows = Vec::new();
        for i in 0..12 {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.horizon_secs = 168 * 3600;
            // First 9 rows are flat (< FLAT_RETURN_EPSILON), last 3 are
            // strongly directional.
            r.final_pred_return = Some(if i < 9 { 1e-10 } else { 0.1 + i as f64 * 0.01 });
            r.final_actual_return = Some(0.02 * (i as f64 - 6.0));
            rows.push(r);
        }
        let agg = aggregate_by_horizon(&rows);
        assert_eq!(agg.len(), 1);
        let h = &agg[0];
        assert_eq!(h.horizon_secs, 168 * 3600);
        assert_eq!(h.n, 12);
        assert_eq!(h.flat_count, 9);
    }

    #[test]
    fn aggregate_by_horizon_partitions_by_horizon() {
        let mut rows = Vec::new();
        for i in 0..3 {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.horizon_secs = 24 * 3600;
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        for i in 0..2 {
            let mut r = base_row(&format!("L{i}"), ts(0), "Trending");
            r.horizon_secs = 168 * 3600;
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let agg = aggregate_by_horizon(&rows);
        assert_eq!(agg.len(), 2);
        // Sorted ascending by horizon_secs.
        assert_eq!(agg[0].horizon_secs, 24 * 3600);
        assert_eq!(agg[0].n, 3);
        assert_eq!(agg[1].horizon_secs, 168 * 3600);
        assert_eq!(agg[1].n, 2);
    }

    #[test]
    fn aggregate_by_horizon_cross_sectional_ic_requires_min_n() {
        let mut rows = Vec::new();
        for i in 0..(CROSS_SECTIONAL_MIN_N - 1) {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.horizon_secs = 24 * 3600;
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let agg = aggregate_by_horizon(&rows);
        assert_eq!(agg[0].cross_sectional_n, CROSS_SECTIONAL_MIN_N - 1);
        assert_eq!(agg[0].cross_sectional_ic, None);
    }

    #[test]
    fn aggregate_by_horizon_cross_sectional_ic_computes_perfect_corr() {
        let mut rows = Vec::new();
        for i in 0..CROSS_SECTIONAL_MIN_N {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.horizon_secs = 24 * 3600;
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let agg = aggregate_by_horizon(&rows);
        let ic = agg[0]
            .cross_sectional_ic
            .expect("ic computable above threshold");
        assert!((ic - 1.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_by_horizon_decile_spread_captures_inversion() {
        // 20 rows: predictions ascending, actuals descending → perfect
        // inverse. Top-decile mean actual should be very negative, bottom
        // very positive, so spread is strongly negative.
        let mut rows = Vec::new();
        for i in 0..20 {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.horizon_secs = 168 * 3600;
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(-(i as f64));
            rows.push(r);
        }
        let agg = aggregate_by_horizon(&rows);
        let spread = agg[0].decile_spread.expect("spread computable");
        assert!(spread < 0.0, "expected negative spread, got {spread}");
    }

    #[test]
    fn aggregate_by_horizon_decile_spread_none_below_ten_rows() {
        let mut rows = Vec::new();
        for i in 0..9 {
            let mut r = base_row(&format!("S{i}"), ts(0), "Trending");
            r.horizon_secs = 24 * 3600;
            r.final_pred_return = Some(i as f64);
            r.final_actual_return = Some(i as f64);
            rows.push(r);
        }
        let agg = aggregate_by_horizon(&rows);
        assert_eq!(agg[0].decile_spread, None);
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
