use chrono::NaiveDateTime;
use common::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

/// Maximum absolute value enforced on the estimated AR(1) coefficient
/// `φ`. Estimates outside the open interval `(-1, 1)` correspond to
/// non-stationary processes whose forecasts grow without bound;
/// clamping the OLS estimate to `±AR1_PHI_CLAMP` keeps the recursion
/// `r_{t+1} = c + φ r_t` strictly convergent (`r → c / (1 − φ)`), so the
/// reverting model always pulls returns toward a finite long-run mean.
pub(crate) const AR1_PHI_CLAMP: f64 = 0.99;

/// Minimum number of observations required to estimate the AR(1)
/// coefficient with enough signal to be useful. Below this threshold
/// the model emits a flat (last-value) forecast rather than gambling on
/// a high-variance estimate. Twenty return pairs (`n = 22` price
/// observations) is the conventional smallest sample where the OLS
/// slope is not dominated by noise.
pub(crate) const AR1_MIN_SAMPLES: usize = 22;

/// AR(1) reverting forecast model.
///
/// Estimates `r_t = c + φ r_{t-1} + ε` on the first-difference series
/// (returns) via OLS, clamps `|φ| < 1`, then projects the horizon
/// recursively. Because `|φ| < 1`, the forecast returns converge to
/// `c / (1 − φ)` (the unconditional mean), which makes the level
/// forecast revert to a constant pace rather than continue the
/// last-observed momentum.
///
/// Designed to complement the momentum-only base ensemble (ETS, Theta,
/// MSTL, NPTS, SeasonalNaive) when the production retro identifies a
/// confident momentum forecast that the cross-section then reverses
/// against — see `improve.md` 2026-06-15 updates 7-8 for the
/// convex-hull argument that motivated adding a structurally
/// mean-reverting base.
pub struct Ar1RevertingModel;

impl Ar1RevertingModel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Ar1RevertingModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ForecastModel for Ar1RevertingModel {
    fn name(&self) -> &str {
        "Ar1Reverting"
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Fast
    }

    fn fit_predict(
        &mut self,
        values: &[f64],
        _timestamps: &[NaiveDateTime],
        horizon: usize,
    ) -> Result<ForecastOutput> {
        let n = values.len();
        if n < 3 {
            return Err(ChronosError::InsufficientData(
                "Ar1Reverting requires at least 3 data points".into(),
            ));
        }

        debug!(
            data_length = n,
            horizon = horizon,
            "Ar1Reverting model fitting"
        );

        let last = *values.last().unwrap();

        if n < AR1_MIN_SAMPLES {
            debug!(
                n,
                threshold = AR1_MIN_SAMPLES,
                "history too short for AR(1) estimation, falling back to flat"
            );
            return Ok(ForecastOutput {
                mean: vec![last; horizon],
                lower_quantile: None,
                upper_quantile: None,
                model_name: "Ar1Reverting".into(),
            });
        }

        let returns: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let m = returns.len();

        let x: &[f64] = &returns[..m - 1];
        let y: &[f64] = &returns[1..];
        let (phi_raw, c) = simple_linreg(x, y);
        let phi = phi_raw.clamp(-AR1_PHI_CLAMP, AR1_PHI_CLAMP);

        debug!(
            phi_raw = phi_raw,
            phi_clamped = phi,
            c = c,
            "Ar1Reverting estimated"
        );

        let last_return = *returns.last().unwrap();
        let mut prev_return = last_return;
        let mut level = last;
        let mut mean = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let r_next = c + phi * prev_return;
            level += r_next;
            mean.push(level);
            prev_return = r_next;
        }

        let fitted: Vec<f64> = x.iter().map(|&r| c + phi * r).collect();
        let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();
        let residual_std = {
            let mean_r = residuals.iter().sum::<f64>() / residuals.len() as f64;
            let var = residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
                / (residuals.len() as f64 - 1.0).max(1.0);
            var.sqrt()
        };

        let z = 1.2816;
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        let mut cum_var = 0.0_f64;
        let mut phi_pow_sq = 1.0_f64;
        for &m in mean.iter() {
            cum_var += phi_pow_sq;
            phi_pow_sq *= phi * phi;
            let band = z * residual_std * cum_var.sqrt();
            lower.push(m - band);
            upper.push(m + band);
        }

        Ok(ForecastOutput {
            mean,
            lower_quantile: Some(lower),
            upper_quantile: Some(upper),
            model_name: "Ar1Reverting".into(),
        })
    }
}

fn simple_linreg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, sum_y / n);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        (0..n)
            .map(|i| base + chrono::Duration::hours(i as i64))
            .collect()
    }

    fn integrate(returns: &[f64], start: f64) -> Vec<f64> {
        let mut v = Vec::with_capacity(returns.len() + 1);
        v.push(start);
        for &r in returns {
            v.push(*v.last().unwrap() + r);
        }
        v
    }

    #[test]
    fn test_ar1_recovers_negative_phi_on_deterministic_series() {
        // r_t = -0.6 * r_{t-1}, integrated to price. The OLS estimate of
        // φ on a noiseless AR(1) recurrence must recover the data
        // generating coefficient exactly (up to numerical precision).
        let target_phi = -0.6_f64;
        let mut returns = vec![1.0_f64];
        for _ in 0..200 {
            let next = target_phi * returns.last().unwrap();
            returns.push(next);
        }
        let values = integrate(&returns, 100.0);
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 5).unwrap();
        assert_eq!(out.mean.len(), 5);

        // The next deterministic return after `returns.last()` is
        // target_phi * returns.last(); the forecast must match it.
        let last_return = *returns.last().unwrap();
        let expected_next = target_phi * last_return;
        let last_value = *values.last().unwrap();
        let predicted_next = out.mean[0];
        assert!(
            (predicted_next - (last_value + expected_next)).abs() < 1e-6,
            "first-step forecast {} should match recurrence {} + {} = {}",
            predicted_next,
            last_value,
            expected_next,
            last_value + expected_next,
        );
    }

    #[test]
    fn test_ar1_forecast_returns_decay_toward_long_run_mean() {
        // For any |φ| < 1 the recursion r_{t+1} = c + φ r_t converges to
        // c / (1 − φ). On a noiseless series with c = 0 the forecast
        // returns must shrink toward 0 as horizon grows, so consecutive
        // forecast levels must converge.
        let mut returns = vec![1.0_f64];
        for _ in 0..200 {
            returns.push(-0.6 * returns.last().unwrap());
        }
        let values = integrate(&returns, 100.0);
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 100).unwrap();
        let last = *values.last().unwrap();

        let early_step = (out.mean[1] - out.mean[0]).abs();
        let late_step = (out.mean[99] - out.mean[98]).abs();
        assert!(
            late_step < early_step + 1e-12,
            "consecutive forecast returns must not grow: early {}, late {}",
            early_step,
            late_step,
        );
        // The level forecast asymptotes; the final point must stay close
        // to the last observation rather than drift away.
        assert!(
            (out.mean[99] - last).abs() < 2.0,
            "long-horizon forecast {} must not drift far from anchor {}",
            out.mean[99],
            last,
        );
    }

    #[test]
    fn test_ar1_falls_back_to_flat_on_short_history() {
        // Fewer than AR1_MIN_SAMPLES observations: the model must not
        // attempt OLS and must return a flat last-value forecast so the
        // pipeline can blend it as a no-op.
        let values: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 5).unwrap();
        let last = *values.last().unwrap();
        for v in &out.mean {
            assert!((v - last).abs() < 1e-12);
        }
        assert!(out.lower_quantile.is_none());
        assert!(out.upper_quantile.is_none());
    }

    #[test]
    fn test_ar1_handles_constant_series() {
        // Constant input → all returns are zero → OLS denominator is
        // zero → the linreg helper falls back to (0, 0), giving a flat
        // forecast at the last observed value with no panics.
        let values = vec![42.0_f64; 50];
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 7).unwrap();
        for v in &out.mean {
            assert!((*v - 42.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ar1_clamps_explosive_estimate() {
        // Manually construct values where the raw OLS estimate would be
        // > 1: a deterministic monotone return sequence with φ_true=1.
        // The clamp must hold |φ| ≤ AR1_PHI_CLAMP < 1, keeping the
        // forecast bounded.
        let mut returns = vec![1.0_f64];
        for _ in 0..100 {
            returns.push(1.05 * returns.last().unwrap());
        }
        let values = integrate(&returns, 100.0);
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 50).unwrap();
        for v in &out.mean {
            assert!(
                v.is_finite(),
                "clamped AR(1) forecast must stay finite, got {}",
                v
            );
        }
    }

    #[test]
    fn test_ar1_insufficient_data() {
        let mut model = Ar1RevertingModel::new();
        let result = model.fit_predict(&[1.0, 2.0], &make_timestamps(2), 3);
        assert!(result.is_err());
    }
}
