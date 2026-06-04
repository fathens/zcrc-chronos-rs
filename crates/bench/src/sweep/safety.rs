//! Input validation and `Result`-returning wrappers around the panic-based
//! metric helpers in `crate::direction_metrics` and `crate::metrics`.
//!
//! The base metric functions intentionally panic on invariant violations
//! (length mismatch, empty input, zero baseline) so that bugs surface
//! loudly in the simple single-shot benchmark path. The sweep driver, by
//! contrast, walks a Cartesian product where many of those preconditions
//! will fail by construction (insufficient actual data after a late
//! `eval_date`, degenerate baseline, etc.). The wrappers in this module
//! validate the inputs up front and convert preventable failures into
//! `SweepError` so the driver can keep evaluating the remaining jobs.

use crate::direction_metrics::{
    CalibrationBucket, DirectionMetrics, compute_calibration_buckets, compute_direction_metrics,
};
use crate::metrics::{MetricSet, compute_metrics};
use crate::sweep::{SweepError, SweepResult};

/// Lower bound on the magnitude of a baseline (current) value, below which
/// `(p - baseline) / baseline` is at risk of producing `Infinity` from
/// finite inputs. Chosen so that the worst-case forecast magnitude bounded
/// by `safe_exp(709) ≈ 8.22e307` cannot inflate the return into `Inf` —
/// `8.22e307 / 1e-150 ≈ 8.22e457`, which still overflows, but anything
/// smaller than this baseline is a degenerate price signal that the
/// predictor was never designed to handle.
pub const BASELINE_MIN_ABS: f64 = 1e-150;

/// Validate that a baseline value is finite and not at or near zero.
///
/// `compute_direction_metrics` divides by `current_value`, so a zero or
/// sub-denormal baseline silently produces `Inf` returns that corrupt the
/// Spearman ranking. This guard rejects those inputs before the metric is
/// computed.
pub fn validate_current_value(value: f64) -> SweepResult<f64> {
    if !value.is_finite() || value.abs() < BASELINE_MIN_ABS {
        return Err(SweepError::DegenerateBaseline { value });
    }
    Ok(value)
}

/// Validate that the magnitude threshold passed to direction metrics is a
/// finite, non-negative number. A NaN threshold silently filters every
/// step (since `x.abs() >= NaN` is always false) and would be reported
/// as "no signal" rather than a configuration error.
pub fn validate_signal_threshold(threshold: f64) -> SweepResult<()> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(SweepError::InvalidConfig(format!(
            "signal_threshold must be finite and non-negative, got {threshold}"
        )));
    }
    Ok(())
}

/// Validate the calibration bucket definition: non-empty, every bucket
/// `low < high`, and all bounds finite.
pub fn validate_calibration_buckets(buckets: &[(f64, f64)]) -> SweepResult<()> {
    if buckets.is_empty() {
        return Err(SweepError::InvalidConfig(
            "calibration_buckets must contain at least one bucket".to_string(),
        ));
    }
    for &(lo, hi) in buckets {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(SweepError::InvalidConfig(format!(
                "calibration bucket [{lo}, {hi}) has non-finite bound"
            )));
        }
        if lo >= hi {
            return Err(SweepError::InvalidConfig(format!(
                "calibration bucket [{lo}, {hi}) requires low < high"
            )));
        }
    }
    Ok(())
}

/// Validate that the forecast and actual vectors are non-empty, equal in
/// length, and that every entry is finite (no NaN / ±Inf). The finiteness
/// check defends against `safe_exp` overflow injecting `Inf` into the
/// forecast when the predictor is asked to extrapolate far beyond the
/// training range.
pub fn validate_forecast_actual_pair(forecast: &[f64], actual: &[f64]) -> SweepResult<()> {
    if forecast.is_empty() {
        return Err(SweepError::InvalidConfig(
            "forecast must not be empty".to_string(),
        ));
    }
    if forecast.len() != actual.len() {
        return Err(SweepError::InvalidConfig(format!(
            "forecast/actual length mismatch: {} vs {}",
            forecast.len(),
            actual.len()
        )));
    }
    if !forecast.iter().all(|v| v.is_finite()) {
        return Err(SweepError::InvalidConfig(
            "forecast contains non-finite value (NaN or Infinity)".to_string(),
        ));
    }
    if !actual.iter().all(|v| v.is_finite()) {
        return Err(SweepError::InvalidConfig(
            "actual contains non-finite value (NaN or Infinity)".to_string(),
        ));
    }
    Ok(())
}

/// Result-returning wrapper around [`compute_direction_metrics`].
///
/// Validates `current_value` (baseline), `signal_threshold`, and the
/// forecast/actual pair, then delegates to the panic-based implementation.
/// All preventable preconditions have been checked, so the inner asserts
/// represent unreachable invariant violations.
pub fn try_compute_direction_metrics(
    forecast: &[f64],
    actual: &[f64],
    current_value: f64,
    signal_threshold: f64,
) -> SweepResult<DirectionMetrics> {
    validate_forecast_actual_pair(forecast, actual)?;
    validate_current_value(current_value)?;
    validate_signal_threshold(signal_threshold)?;
    Ok(compute_direction_metrics(
        forecast,
        actual,
        current_value,
        signal_threshold,
    ))
}

/// Result-returning wrapper around [`compute_calibration_buckets`].
pub fn try_compute_calibration_buckets(
    forecast: &[f64],
    actual: &[f64],
    current_value: f64,
    buckets: &[(f64, f64)],
) -> SweepResult<Vec<CalibrationBucket>> {
    validate_forecast_actual_pair(forecast, actual)?;
    validate_current_value(current_value)?;
    validate_calibration_buckets(buckets)?;
    Ok(compute_calibration_buckets(
        forecast,
        actual,
        current_value,
        buckets,
    ))
}

/// Result-returning wrapper around [`compute_metrics`].
///
/// Adds the same pair-validation as the direction wrapper. `train_values`
/// is checked for emptiness because `compute_metrics` would otherwise
/// divide by zero when scaling MASE.
pub fn try_compute_metrics(
    forecast: &[f64],
    actual: &[f64],
    train_values: &[f64],
    season: usize,
) -> SweepResult<MetricSet> {
    validate_forecast_actual_pair(forecast, actual)?;
    if train_values.is_empty() {
        return Err(SweepError::InvalidConfig(
            "train_values must not be empty".to_string(),
        ));
    }
    if season == 0 {
        return Err(SweepError::InvalidConfig(
            "season must be at least 1".to_string(),
        ));
    }
    Ok(compute_metrics(forecast, actual, train_values, season))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_baseline() {
        assert!(matches!(
            validate_current_value(0.0),
            Err(SweepError::DegenerateBaseline { .. })
        ));
    }

    #[test]
    fn rejects_subnormal_baseline() {
        // 1e-200 is finite but below BASELINE_MIN_ABS, so it must be
        // rejected to avoid producing Inf returns.
        assert!(matches!(
            validate_current_value(1e-200),
            Err(SweepError::DegenerateBaseline { .. })
        ));
    }

    #[test]
    fn accepts_normal_baseline() {
        assert_eq!(validate_current_value(1.0).unwrap(), 1.0);
        assert_eq!(validate_current_value(1e-100).unwrap(), 1e-100);
    }

    #[test]
    fn rejects_nan_baseline() {
        assert!(matches!(
            validate_current_value(f64::NAN),
            Err(SweepError::DegenerateBaseline { .. })
        ));
    }

    #[test]
    fn rejects_nan_threshold() {
        assert!(matches!(
            validate_signal_threshold(f64::NAN),
            Err(SweepError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_negative_threshold() {
        assert!(matches!(
            validate_signal_threshold(-0.01),
            Err(SweepError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_inverted_calibration_bucket() {
        assert!(matches!(
            validate_calibration_buckets(&[(0.5, 0.1)]),
            Err(SweepError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_empty_calibration_buckets() {
        assert!(matches!(
            validate_calibration_buckets(&[]),
            Err(SweepError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_non_finite_forecast() {
        let forecast = [1.0, f64::INFINITY];
        let actual = [1.0, 1.5];
        assert!(matches!(
            validate_forecast_actual_pair(&forecast, &actual),
            Err(SweepError::InvalidConfig(_))
        ));
    }

    #[test]
    fn try_compute_direction_metrics_propagates_validation() {
        let forecast = [100.5, 101.0, 102.0];
        let actual = [100.4, 100.9, 102.1];
        // Zero baseline must be rejected before calling the inner panic.
        assert!(matches!(
            try_compute_direction_metrics(&forecast, &actual, 0.0, 0.005),
            Err(SweepError::DegenerateBaseline { .. })
        ));
        // Valid inputs flow through to the inner function and produce
        // a DirectionMetrics with Some(ic).
        let metrics = try_compute_direction_metrics(&forecast, &actual, 100.0, 0.005).unwrap();
        assert!(metrics.ic.is_some());
    }

    #[test]
    fn try_compute_metrics_rejects_empty_train() {
        let forecast = [1.0];
        let actual = [1.1];
        let train: [f64; 0] = [];
        assert!(matches!(
            try_compute_metrics(&forecast, &actual, &train, 1),
            Err(SweepError::InvalidConfig(_))
        ));
    }
}
