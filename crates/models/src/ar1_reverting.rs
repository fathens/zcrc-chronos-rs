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

        // Cumulative AR(1) level variance: the h-step level forecast is
        // the running sum of returns, each of which has a damped error
        // contribution `Σᵢ₌₀ˢ⁻¹ φⁱ = (1 − φˢ) / (1 − φ)`. Squaring and
        // summing those weights gives `Var(level_h) = σ² · Σₛ₌₁ʰ
        // ((1 − φˢ) / (1 − φ))²`, which grows like √h as `φ → 0` (random
        // walk) and asymptotes for `|φ|` close to 1. The earlier
        // `Σⱼ φ²ʲ` form underestimated this — it bounded the band even
        // when the underlying process was a random walk.
        let z = 1.2816;
        let inv = 1.0 / (1.0 - phi);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        let mut cum_var = 0.0_f64;
        let mut phi_s = phi;
        for &m in mean.iter() {
            let w = (1.0 - phi_s) * inv;
            cum_var += w * w;
            let band = z * residual_std * cum_var.sqrt();
            lower.push(m - band);
            upper.push(m + band);
            phi_s *= phi;
        }

        Ok(ForecastOutput {
            mean,
            lower_quantile: Some(lower),
            upper_quantile: Some(upper),
            model_name: "Ar1Reverting".into(),
        })
    }
}

/// Centered ordinary least squares: returns `(slope, intercept)` for
/// the linear fit `y = slope * x + intercept`.
///
/// The centered normal equations `Sxx = Σ(xᵢ − x̄)²`,
/// `Sxy = Σ(xᵢ − x̄)(yᵢ − ȳ)` avoid the catastrophic cancellation that
/// the textbook form `n·Σx² − (Σx)²` suffers from on data with a large
/// non-zero mean. The Ar1 reverting model fits returns (typically
/// `O(1)` or smaller), but the helper guards against a later caller
/// passing raw price-scale samples where `Σx² ≈ (Σx)²` falls inside the
/// ULP of `f64`, which would turn the slope estimate into noise and let
/// the AR(1) recursion blow the level forecast up via an arbitrarily
/// large intercept `c`.
fn simple_linreg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        sxx += dx * dx;
        sxy += dx * (yi - mean_y);
    }

    if sxx < 1e-15 {
        return (0.0, mean_y);
    }

    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    if !slope.is_finite() || !intercept.is_finite() {
        return (0.0, mean_y);
    }
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

    #[test]
    fn test_ar1_band_grows_with_horizon_when_phi_is_near_zero() {
        // Build a near-random-walk: independent returns drawn from a
        // simple deterministic shuffle so phi estimates close to zero
        // and the residual variance dominates. The level forecast then
        // accumulates uncorrelated step errors, so the band must grow
        // like √h. The earlier `Σⱼ φ²ʲ` formula collapsed to 1 for
        // φ → 0 and held the band constant — that's the bug fixed
        // here.
        let mut returns = Vec::with_capacity(200);
        for i in 0..200 {
            // Alternating +1 / −1 returns make Σ correlations vanish
            // and phi ≈ −1 (perfectly anti-correlated). To stay near
            // phi ≈ 0 instead, use a longer-period pattern that breaks
            // the lag-1 correlation: 1, 1, −1, −1 repeating.
            let v = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
            returns.push(v);
        }
        let values = integrate(&returns, 100.0);
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 50).unwrap();
        let lower = out.lower_quantile.expect("band must be Some");
        let upper = out.upper_quantile.expect("band must be Some");
        let band_at = |h: usize| -> f64 { upper[h] - lower[h] };

        let early = band_at(0);
        let late = band_at(49);
        assert!(
            late > early * 3.0,
            "band must grow with horizon when phi is small; early {} late {}",
            early,
            late,
        );
    }

    #[test]
    fn test_ar1_band_collapses_to_zero_on_constant_series() {
        // Constant input has no residual variance, so the band must be
        // exactly zero at every horizon. This guards the band formula
        // against producing a spurious non-zero band when residual_std
        // collapses.
        let values = vec![42.0_f64; 50];
        let ts = make_timestamps(values.len());
        let mut model = Ar1RevertingModel::new();
        let out = model.fit_predict(&values, &ts, 5).unwrap();
        let lower = out.lower_quantile.expect("band must be Some");
        let upper = out.upper_quantile.expect("band must be Some");
        for (lo, up) in lower.iter().zip(upper.iter()) {
            assert!(
                (up - lo).abs() < 1e-9,
                "band must vanish; got [{}, {}]",
                lo,
                up
            );
        }
    }

    #[test]
    fn test_ar1_linreg_stable_under_large_scale_offset() {
        // The classical `n·Σx² − (Σx)²` form of OLS underflows when the
        // sample mean is large compared to the spread: both terms grow
        // like (n·x̄)² while their difference is the desired
        // `n·Σ(x−x̄)²`. Constructing an explicit case where the textbook
        // form returns garbage but the centered form recovers the true
        // slope locks the centering in.
        let n = 100;
        let offset = 1.0e9_f64;
        // True slope = 0.001, intercept after shift = offset.
        // y = 0.001 * x + offset (no noise).
        let xs: Vec<f64> = (0..n).map(|i| offset + i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 0.001 * x + offset).collect();
        let (slope, intercept) = simple_linreg(&xs, &ys);
        assert!(
            (slope - 0.001).abs() < 1e-6,
            "centered slope must recover the true value; got {}",
            slope,
        );
        // Intercept is huge (≈ 1.001e9), only assert it stays finite
        // and the residual `y - (slope*x + intercept)` is tiny.
        assert!(intercept.is_finite());
        let resid_max = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| (y - (slope * x + intercept)).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            resid_max < 1e-3,
            "fit residual must be small under large-scale offset; got {}",
            resid_max,
        );
    }
}
