//! Horizon-dependent magnitude shrinkage for long-horizon forecasts.
//!
//! Applied at the output of [`crate::npts::NptsModel`] and the
//! `(linear + theta2) / 2` blend of [`crate::theta::ThetaModel`]. Both
//! models produce confident, high-magnitude predictions over multi-day
//! horizons that the production diagnostic
//! (`crates/predictor/tests/real_data_over_damping.rs`,
//! `top_decile_decomposition`) identified as the dominant source of the
//! "high-confidence wrong-direction" tail.
//!
//! Production fixtures (1084 NEAR-token snapshots, h = 168 h):
//! - TOP-decile DirAcc = 28.1 % (96 evaluations)
//! - TOP-decile mean actual return = -5.42 %
//! - TOP-decile puller: Theta 47/108, MstlEts 34/108, NPTS 27/108
//! - **Top-15 by predicted magnitude**: NPTS 7/15, MstlEts 5/15, Theta 3/15
//!
//! Markowitz / Top-N selection weighs the magnitude tail, so the extreme
//! values dominate portfolio losses even when count-wise Theta is more
//! frequent. The shrinkage attenuates the deviation of each forecast
//! step from the last observed value:
//!
//! ```text
//! forecast[h] := last + exp(-h / τ) × (raw[h] − last)
//! ```
//!
//! - At `h = 0` (one step ahead) the shrinkage is `exp(-1 / τ) ≈ 1`, so
//!   short-horizon predictions are essentially untouched.
//! - At `h = 168` (zaciraci production horizon) and `τ = 72` the factor
//!   is `exp(-169 / 72) ≈ 0.097`, attenuating long-horizon magnitudes by
//!   ~10×. A confident +35 % NPTS prediction shrinks to ~+3.5 %; an
//!   apys-style Theta linear blow-up of +8.7 % shrinks to ~+0.86 %.
//!
//! `h0` and τ are intentionally exposed as `pub(crate)` constants so the
//! production team can grid-search them via a patched build until
//! `predict_sweep` decile_spread crosses zero at the production horizon.

/// First horizon step subject to [`apply_long_horizon_shrinkage`], in
/// forecast-step units. Steps with `h < LONG_HORIZON_SHRINK_H0` are left
/// untouched so that the bench fixtures and other short-horizon
/// (≤ 24 steps ≈ one day for hourly data) callers do not lose the
/// magnitude that NPTS / Theta correctly carry on predictable series.
pub(crate) const LONG_HORIZON_SHRINK_H0: usize = 24;

/// Time constant for [`apply_long_horizon_shrinkage`], in forecast-step
/// units. With `h0 = 24` and `tau = 72`:
/// - h = 24 (one day in hourly data): shrink = 1.000 (untouched)
/// - h = 72 (three days): shrink = exp(-48/72) ≈ 0.513
/// - h = 168 (one week, zaciraci production horizon): shrink ≈ 0.135
///   (7× attenuation)
pub(crate) const LONG_HORIZON_SHRINK_TAU: f64 = 72.0;

/// Shrink each forecast step exponentially toward `last`, in place.
///
/// `forecast[h]` is left unchanged when `h < h0`. For
/// `h ≥ h0` it is replaced by
/// `last + exp(-(h − h0 + 1) / tau) × (forecast[h] − last)`.
///
/// `tau` must be positive; values ≤ 0 panic.
pub(crate) fn apply_long_horizon_shrinkage(forecast: &mut [f64], last: f64, tau: f64, h0: usize) {
    assert!(
        tau > 0.0 && tau.is_finite(),
        "shrinkage tau must be positive and finite, got {tau}"
    );
    if !last.is_finite() {
        // Refuse to shrink against a non-finite baseline — leave the raw
        // forecast intact so the caller's downstream defenses (NaN/Inf
        // checks in safety.rs) catch the upstream problem.
        return;
    }
    for (h, m) in forecast.iter_mut().enumerate() {
        if h < h0 || !m.is_finite() {
            continue;
        }
        let shrink = (-((h - h0 + 1) as f64) / tau).exp();
        *m = last + shrink * (*m - last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrinkage_pulls_toward_last_beyond_h0() {
        let last = 100.0;
        // Raw forecast is +50 % for every step. Steps below h0 must stay
        // at 150; steps above h0 must decay monotonically toward `last`.
        let mut forecast = vec![150.0; 200];
        apply_long_horizon_shrinkage(&mut forecast, last, 72.0, 24);
        // h < 24: untouched.
        for v in &forecast[..24] {
            assert!((v - 150.0).abs() < 1e-12);
        }
        // h = 24 (boundary): shrink = exp(-1/72) ≈ 0.986
        assert!((forecast[24] - (100.0 + 0.986 * 50.0)).abs() < 0.2);
        // h = 95 (24 + τ): shrink = exp(-72/72) ≈ 0.368
        assert!((forecast[95] - (100.0 + 0.368 * 50.0)).abs() < 0.2);
        // h = 168 (production horizon): shrink ≈ exp(-145/72) ≈ 0.1338
        assert!(forecast[168] < 107.0);
        assert!(forecast[168] > 105.0);
        // Monotonicity toward `last` beyond h0.
        for window in forecast[24..].windows(2) {
            assert!(window[0] > window[1], "shrinkage must be monotone");
            assert!(window[1] > last, "must not cross `last` for positive raw");
        }
    }

    #[test]
    fn shrinkage_preserves_baseline_when_raw_equals_last() {
        let last = 1.5;
        let mut forecast = vec![1.5; 50];
        apply_long_horizon_shrinkage(&mut forecast, last, 72.0, 24);
        for v in &forecast {
            assert!((v - last).abs() < 1e-12);
        }
    }

    #[test]
    fn shrinkage_reduces_negative_deviation_too() {
        let last = 100.0;
        let mut forecast = vec![50.0; 200]; // raw deviation = -50
        apply_long_horizon_shrinkage(&mut forecast, last, 72.0, 24);
        for v in &forecast[..24] {
            assert!((v - 50.0).abs() < 1e-12);
        }
        for v in &forecast[24..] {
            assert!(*v < last, "negative deviation must stay below last");
            assert!(*v > 50.0, "negative deviation must shrink toward last");
        }
    }

    #[test]
    fn shrinkage_is_no_op_when_h0_exceeds_horizon() {
        let last = 100.0;
        let raw = vec![150.0; 10];
        let mut forecast = raw.clone();
        apply_long_horizon_shrinkage(&mut forecast, last, 72.0, 24);
        assert_eq!(forecast, raw, "h0 > horizon must leave forecast intact");
    }

    #[test]
    fn shrinkage_leaves_baseline_alone_when_last_non_finite() {
        let raw = vec![50.0; 100];
        let mut forecast = raw.clone();
        apply_long_horizon_shrinkage(&mut forecast, f64::NAN, 72.0, 24);
        assert_eq!(forecast, raw);
    }

    #[test]
    fn shrinkage_skips_non_finite_forecast_entries() {
        let last = 10.0;
        let mut forecast = vec![20.0; 30];
        forecast[25] = f64::INFINITY;
        apply_long_horizon_shrinkage(&mut forecast, last, 72.0, 24);
        // h < 24 untouched.
        assert!((forecast[0] - 20.0).abs() < 1e-12);
        // h = 24: shrunk.
        assert!(forecast[24].is_finite() && forecast[24] < 20.0);
        // h = 25: Inf left alone.
        assert!(forecast[25].is_infinite());
        // h = 26+: shrunk.
        assert!(forecast[26].is_finite() && forecast[26] < 20.0);
    }

    #[test]
    #[should_panic(expected = "tau must be positive")]
    fn shrinkage_panics_on_non_positive_tau() {
        let mut forecast = vec![1.0; 5];
        apply_long_horizon_shrinkage(&mut forecast, 1.0, 0.0, 24);
    }
}
