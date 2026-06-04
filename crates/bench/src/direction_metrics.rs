//! Direction-based forecast evaluation metrics.
//!
//! Standard magnitude metrics (MAE, MAPE, RMSE) measure how far the forecast
//! is from the actual value, but they do not capture whether the forecast
//! got the *direction* of return right. For trading applications, direction
//! and rank order matter more than magnitude error: a 1% MAPE forecast that
//! consistently points the wrong way produces losses, while a 5% MAPE
//! forecast with the right direction can be profitable.
//!
//! This module provides:
//! - **DirAcc** — fraction of forecasts that match the sign of the actual return.
//! - **DirAcc (filtered)** — DirAcc on the subset where both predicted and
//!   actual returns exceed a magnitude threshold (excludes flat predictions).
//! - **IC (Information Coefficient)** — Spearman rank correlation between
//!   predicted and actual returns. The quant industry standard for forecast
//!   skill.
//! - **Calibration residual** — median of `predicted_return - actual_return`.
//!   Detects systematic bias (e.g. "predicts up but actually goes down").
//! - **Calibration buckets** — `actual` distribution per `predicted` bucket,
//!   used to detect asymmetric calibration (e.g. bearish predictions accurate,
//!   bullish predictions inverted).

use std::cmp::Ordering;

/// Aggregate direction-based metrics over a forecast horizon.
#[derive(Debug, Clone)]
pub struct DirectionMetrics {
    /// Fraction of forecast steps where `sign(pred_return) == sign(actual_return)`.
    /// Includes zero-return steps (sign==0) as a match only when both are zero.
    /// Range: 0.0 to 1.0. Random baseline ≈ 0.5.
    pub dir_acc: f64,
    /// DirAcc on the subset of forecasts where both `|pred_return|` and
    /// `|actual_return|` exceed `signal_threshold`. `None` when no forecast
    /// passes the filter.
    pub dir_acc_filtered: Option<f64>,
    /// Number of forecast steps that passed the magnitude filter.
    pub filtered_count: usize,
    /// Total number of forecast steps evaluated.
    pub total_count: usize,
    /// Spearman rank correlation between predicted and actual returns.
    /// Range: -1.0 to 1.0. A positive IC means the forecast's ranking of
    /// returns matches reality. `None` when either series has zero variance
    /// or fewer than two observations — distinguishes "no ranking signal" from
    /// "true zero correlation".
    pub ic: Option<f64>,
    /// Median of `predicted_return - actual_return`. Positive values indicate
    /// the forecast is biased upward (predicts more than actual).
    pub calibration_residual: f64,
}

/// One bucket of the predicted-return distribution, with statistics on the
/// corresponding actual returns. Used to detect asymmetric calibration.
#[derive(Debug, Clone)]
pub struct CalibrationBucket {
    /// Human-readable label, e.g. "[-5%, -1%)".
    pub label: String,
    /// Lower bound of the predicted-return range (inclusive).
    pub pred_return_low: f64,
    /// Upper bound of the predicted-return range (exclusive).
    pub pred_return_high: f64,
    /// Median of actual returns whose predicted return fell in this bucket.
    pub actual_return_median: f64,
    /// Fraction of bucket members whose actual return was positive.
    pub actual_positive_rate: f64,
    /// Number of forecasts in this bucket.
    pub count: usize,
}

/// Compute direction-based metrics for a forecast.
///
/// # Arguments
/// * `forecast` — predicted values at each forecast step.
/// * `actual` — actual values at each forecast step (same length as `forecast`).
/// * `current_value` — last observed training value, used as the baseline
///   to compute returns. Must be non-zero.
/// * `signal_threshold` — magnitude threshold for the filtered DirAcc, in
///   return units (e.g. 0.005 = 0.5%). A common choice from the predict_sweep
///   evaluation is 0.005.
///
/// # Panics
/// Panics if `forecast` and `actual` have different lengths, are empty, or
/// if `current_value` is zero.
pub fn compute_direction_metrics(
    forecast: &[f64],
    actual: &[f64],
    current_value: f64,
    signal_threshold: f64,
) -> DirectionMetrics {
    assert_eq!(
        forecast.len(),
        actual.len(),
        "forecast and actual must have the same length"
    );
    assert!(!forecast.is_empty(), "forecast must not be empty");
    assert!(current_value != 0.0, "current_value must not be zero");

    let pred_returns: Vec<f64> = forecast
        .iter()
        .map(|p| (p - current_value) / current_value)
        .collect();
    let actual_returns: Vec<f64> = actual
        .iter()
        .map(|a| (a - current_value) / current_value)
        .collect();

    let total = pred_returns.len();

    // DirAcc: matched signs over all steps.
    let matches = pred_returns
        .iter()
        .zip(&actual_returns)
        .filter(|(p, a)| sign(**p) == sign(**a))
        .count();
    let dir_acc = matches as f64 / total as f64;

    // DirAcc filtered by magnitude threshold.
    let mut filtered_matches = 0usize;
    let mut filtered_total = 0usize;
    for (p, a) in pred_returns.iter().zip(&actual_returns) {
        if p.abs() >= signal_threshold && a.abs() >= signal_threshold {
            filtered_total += 1;
            if sign(*p) == sign(*a) {
                filtered_matches += 1;
            }
        }
    }
    let dir_acc_filtered = if filtered_total > 0 {
        Some(filtered_matches as f64 / filtered_total as f64)
    } else {
        None
    };

    // Information Coefficient — Spearman correlation.
    let ic = spearman_correlation(&pred_returns, &actual_returns);

    // Calibration residual — median of pred - actual return.
    let mut residuals: Vec<f64> = pred_returns
        .iter()
        .zip(&actual_returns)
        .map(|(p, a)| p - a)
        .collect();
    let calibration_residual = median_in_place(&mut residuals);

    DirectionMetrics {
        dir_acc,
        dir_acc_filtered,
        filtered_count: filtered_total,
        total_count: total,
        ic,
        calibration_residual,
    }
}

/// Default predicted-return buckets for calibration analysis, matching the
/// buckets used in zaciraci's predict_sweep evaluation.
pub fn default_calibration_buckets() -> Vec<(f64, f64)> {
    vec![
        (-1.0, -0.05),
        (-0.05, -0.01),
        (-0.01, -0.005),
        (-0.005, 0.005),
        (0.005, 0.01),
        (0.01, 0.05),
        (0.05, 1.0),
    ]
}

/// Bucket forecasts by their predicted return and report the actual-return
/// distribution within each bucket.
///
/// Bucket boundaries are `[low, high)`. Forecasts whose predicted return
/// does not fall in any bucket are silently dropped.
pub fn compute_calibration_buckets(
    forecast: &[f64],
    actual: &[f64],
    current_value: f64,
    buckets: &[(f64, f64)],
) -> Vec<CalibrationBucket> {
    assert_eq!(forecast.len(), actual.len());
    assert!(current_value != 0.0);

    buckets
        .iter()
        .map(|&(low, high)| {
            let mut actuals_in_bucket: Vec<f64> = Vec::new();
            for (p, a) in forecast.iter().zip(actual) {
                let pr = (p - current_value) / current_value;
                if pr >= low && pr < high {
                    let ar = (a - current_value) / current_value;
                    actuals_in_bucket.push(ar);
                }
            }
            let count = actuals_in_bucket.len();
            let (median, positive_rate) = if count > 0 {
                let positive = actuals_in_bucket.iter().filter(|&&v| v > 0.0).count();
                let pr = positive as f64 / count as f64;
                (median_in_place(&mut actuals_in_bucket), pr)
            } else {
                (0.0, 0.0)
            };
            CalibrationBucket {
                label: format!("[{:.1}%, {:.1}%)", low * 100.0, high * 100.0),
                pred_return_low: low,
                pred_return_high: high,
                actual_return_median: median,
                actual_positive_rate: positive_rate,
                count,
            }
        })
        .collect()
}

/// Spearman rank correlation between two equal-length series.
/// Returns `None` when fewer than two observations or either series has zero
/// variance — distinguishes "no ranking signal" from "true zero correlation".
pub(crate) fn spearman_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    assert_eq!(x.len(), y.len());
    if x.len() < 2 {
        return None;
    }
    let x_ranks = compute_ranks(x);
    let y_ranks = compute_ranks(y);
    pearson_correlation(&x_ranks, &y_ranks)
}

/// Pearson correlation. Returns `None` when either series has zero variance.
fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (xi, yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-30 {
        None
    } else {
        Some(cov / denom)
    }
}

/// Average-rank tie handling (matches scipy.stats.rankdata default).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // 1-indexed average rank of positions i..j.
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for entry in &indexed[i..j] {
            ranks[entry.0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn sign(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn median_in_place(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len();
    let mid = n / 2;
    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if n.is_multiple_of(2) {
        let upper = values[mid];
        let lower = values[..mid]
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(upper);
        (lower + upper) / 2.0
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_upward_direction() {
        let current = 100.0;
        let forecast = vec![101.0, 102.0, 103.0, 104.0, 105.0];
        let actual = vec![100.5, 101.5, 102.5, 103.5, 104.5];
        let m = compute_direction_metrics(&forecast, &actual, current, 0.001);
        assert_eq!(m.dir_acc, 1.0);
        assert_eq!(m.dir_acc_filtered, Some(1.0));
        let ic =
            m.ic.expect("ic should be Some when both series have variance");
        assert!(ic > 0.99, "expected near-perfect IC, got {ic}");
    }

    #[test]
    fn test_inverted_direction() {
        let current = 100.0;
        // Predict up, actually down — all forecasts wrong-direction
        let forecast = vec![101.0, 102.0, 103.0];
        let actual = vec![99.0, 98.0, 97.0];
        let m = compute_direction_metrics(&forecast, &actual, current, 0.001);
        assert_eq!(m.dir_acc, 0.0);
        assert_eq!(m.dir_acc_filtered, Some(0.0));
        // IC: monotonically increasing pred + monotonically decreasing actual
        // → ranks are perfectly inverted → IC = -1.
        let ic =
            m.ic.expect("ic should be Some when ranks are non-degenerate");
        assert!((ic - (-1.0)).abs() < 1e-9, "expected IC ≈ -1, got {ic}");
    }

    #[test]
    fn test_filtered_dir_acc_excludes_flat() {
        let current = 100.0;
        // First two: |pred| and |actual| above threshold, both correct direction.
        // Last two: predictions below threshold (flat) — excluded from filtered count.
        let forecast = vec![102.0, 103.0, 100.1, 100.05];
        let actual = vec![101.0, 102.0, 100.5, 100.3];
        let m = compute_direction_metrics(&forecast, &actual, current, 0.005);
        // filtered: 2 entries pass, 2 match = 1.0
        assert_eq!(m.filtered_count, 2);
        assert_eq!(m.dir_acc_filtered, Some(1.0));
        // unfiltered: 4 entries, all sign-matched → 1.0
        assert_eq!(m.dir_acc, 1.0);
    }

    #[test]
    fn test_calibration_residual_detects_bias() {
        let current = 100.0;
        // Predict 5% up, actually 1% up → residual = 4%
        let forecast = vec![105.0, 106.0, 105.0];
        let actual = vec![101.0, 102.0, 101.0];
        let m = compute_direction_metrics(&forecast, &actual, current, 0.001);
        assert!(
            (m.calibration_residual - 0.04).abs() < 1e-9,
            "expected residual ≈ 4%, got {}",
            m.calibration_residual
        );
    }

    #[test]
    fn test_calibration_buckets_capture_asymmetric_bias() {
        let current = 100.0;
        let buckets = default_calibration_buckets();
        // Predictions in +5..+100% bucket, but actuals are negative — captures
        // the "predict up but actually down" pattern from improve.md.
        let forecast = vec![110.0, 115.0, 120.0];
        let actual = vec![98.0, 97.0, 99.0];
        let cb = compute_calibration_buckets(&forecast, &actual, current, &buckets);
        let bullish = cb
            .iter()
            .find(|b| b.pred_return_low == 0.05)
            .expect("bullish bucket missing");
        assert_eq!(bullish.count, 3);
        assert!(
            bullish.actual_return_median < 0.0,
            "bullish predictions had negative actual median: {}",
            bullish.actual_return_median
        );
        assert_eq!(bullish.actual_positive_rate, 0.0);
    }

    #[test]
    fn test_ic_with_random_pred_is_near_zero() {
        let current = 100.0;
        let forecast: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let actual: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.7).cos()).collect();
        let m = compute_direction_metrics(&forecast, &actual, current, 0.001);
        // No relationship → low |IC|
        let ic = m.ic.expect("ic should be Some when series have variance");
        assert!(ic.abs() < 0.5, "expected near-zero IC, got {ic}");
    }

    #[test]
    fn test_compute_ranks_handles_ties() {
        // Values 1, 2, 2, 3 → ranks 1, 2.5, 2.5, 4
        let ranks = compute_ranks(&[1.0, 2.0, 2.0, 3.0]);
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn test_ic_is_none_when_forecast_has_zero_variance() {
        let current = 100.0;
        // All forecasts equal → pred_returns has zero variance → ranking
        // signal is undefined.
        let forecast = vec![100.5, 100.5, 100.5];
        let actual = vec![100.5, 101.0, 99.5];
        let m = compute_direction_metrics(&forecast, &actual, current, 0.001);
        assert!(m.ic.is_none(), "expected None ic, got {:?}", m.ic);
    }

    #[test]
    fn test_dir_acc_filtered_none_when_all_flat() {
        let current = 100.0;
        let forecast = vec![100.001, 100.002, 100.001];
        let actual = vec![100.001, 100.002, 100.001];
        let m = compute_direction_metrics(&forecast, &actual, current, 0.005);
        assert!(m.dir_acc_filtered.is_none());
        assert_eq!(m.filtered_count, 0);
    }
}
