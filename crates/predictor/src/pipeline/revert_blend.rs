//! Post-process layer that optionally pulls a confident momentum
//! forecast back toward an explicit AR(1) reverting prediction.
//!
//! The pipeline's ensemble is a softmax-weighted convex combination of
//! momentum/persistence base models (ETS, Theta, MSTL, NPTS,
//! SeasonalNaive). On a mean-reverting cross-section the ensemble
//! cannot escape the convex hull of those predictions, so a confident
//! up-forecast that the market then reverses produces the
//! confident-wrong tail measured in `improve.md` 2026-06-15 updates
//! 7-8 (pred > +5 % mean actual_return = -2.87 % on n = 162).
//!
//! This module adds a thin overlay that:
//!
//! 1. Computes the post-retrend forecast's predicted simple return
//!    against the **same-domain** anchor (log-return or
//!    price-relative). The anchor is the last observation of the
//!    series fed to the base models, *not* the detrended residual
//!    (which is approximately zero by construction and would explode
//!    the gate).
//!
//! 2. Gates on `horizon ≥ H₀` **and** `|pred_return| > θ` so that only
//!    long-horizon, confident-magnitude forecasts trigger the overlay.
//!    Short-horizon and near-flat predictions pass through unchanged.
//!
//! 3. Fits an `Ar1RevertingModel` on the **trend-included**
//!    same-domain history (`log_values` when the pipeline took a log
//!    transform, otherwise `norm_values`), then blends per-step
//!    `(1 − α) · base + α · ar1` for mean and (when both sides have
//!    bands) lower/upper.
//!
//! The function is intentionally infallible from the caller's
//! perspective: it returns `Option<BlendedForecast>` so any
//! degeneracy (AR(1) failure, NaN, length mismatch, lost band
//! covariance, broken `lower ≤ mean ≤ upper`) collapses to "skip
//! blend, keep base forecast" rather than failing the whole
//! `predict()` call.

use chrono::NaiveDateTime;
use common::ForecastModel;
use models::Ar1RevertingModel;

/// Floor applied to `|anchor|` in the price-domain gate to keep the
/// simple-return computation finite when a token's nominal price
/// underflows toward zero. Matches the `1e-150` floor the detrend gate
/// uses for the same reason.
const REVERT_PRICE_ANCHOR_FLOOR: f64 = 1e-150;

/// Result of a successful blend. `lower` / `upper` mirror whatever
/// covariance the base + AR(1) forecasts could carry: both `Some` when
/// both sides supplied a band, both `None` when the base had no band
/// to begin with. A configuration that strips covariance (base
/// supplied a band but AR(1) did not) is rejected by
/// [`apply_revert_blend`] before this struct is constructed, because
/// blending the mean while keeping the base's bands would break the
/// `lower ≤ mean ≤ upper` invariant.
pub(crate) struct BlendedForecast {
    pub(crate) mean: Vec<f64>,
    pub(crate) lower: Option<Vec<f64>>,
    pub(crate) upper: Option<Vec<f64>>,
    /// The simple-return magnitude that triggered the blend. Surfaced
    /// purely so the pipeline can log it.
    pub(crate) pred_return: f64,
}

/// Apply the reverting blend overlay. Returns `Some` when the blend
/// fired *and* produced a numerically-sound result, `None` otherwise.
///
/// Pure function: the caller is responsible for reading the env-var
/// configuration (alpha / magnitude threshold / horizon threshold)
/// and for the domain selection (`fit_values`, `anchor`,
/// `log_transformed`). Keeping the env reads out lets the module be
/// unit-tested without touching process state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_revert_blend(
    forecast_mean: &[f64],
    forecast_lower: Option<&[f64]>,
    forecast_upper: Option<&[f64]>,
    fit_values: &[f64],
    fit_timestamps: &[NaiveDateTime],
    anchor: f64,
    log_transformed: bool,
    alpha: f64,
    magnitude_threshold: f64,
) -> Option<BlendedForecast> {
    // The blend is opt-in. Treating `α = 0` exactly the same as
    // `α` out of range keeps the default-off behaviour identical to
    // an unset env var.
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return None;
    }
    if !magnitude_threshold.is_finite() || magnitude_threshold < 0.0 {
        return None;
    }
    if forecast_mean.is_empty() {
        return None;
    }
    if !anchor.is_finite() {
        return None;
    }

    let horizon = forecast_mean.len();
    let last_mean = *forecast_mean.last()?;
    if !last_mean.is_finite() {
        return None;
    }

    // Gate: simple return magnitude vs threshold. The two domain forms
    // produce numerically-comparable values: log-return → simple via
    // `exp(Δ) − 1`, price-domain → simple via division. Both fall in
    // the same units as zaciraci's `predict_sweep` filter.
    let pred_return = if log_transformed {
        (last_mean - anchor).exp() - 1.0
    } else {
        let denom = anchor.abs().max(REVERT_PRICE_ANCHOR_FLOOR);
        (last_mean - anchor) / denom
    };
    if !pred_return.is_finite() {
        return None;
    }
    if pred_return.abs() <= magnitude_threshold {
        return None;
    }

    // Fit AR(1) on the same-domain history. `Ar1RevertingModel`
    // already handles its own degenerate cases (short history flat
    // fallback, constant series, explosive φ clamp).
    let mut model = Ar1RevertingModel::new();
    let ar1 = model
        .fit_predict(fit_values, fit_timestamps, horizon)
        .ok()?;

    if ar1.mean.len() != horizon {
        return None;
    }
    if ar1.mean.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Decide the band-blend rule based on the four possible
    // (base band, ar1 band) cases. The hard case is "base has a band
    // but AR(1) doesn't" (e.g. AR(1) fell back to flat on short
    // history): keeping the base's band on a blended mean would let
    // `mean` drift outside `[lower, upper]`, so we skip the entire
    // blend instead. See implementation-planning Phase 2 minutes for
    // the rationale.
    let blend_band = match (
        forecast_lower,
        forecast_upper,
        ar1.lower_quantile.as_ref(),
        ar1.upper_quantile.as_ref(),
    ) {
        (Some(_), Some(_), Some(al), Some(au)) => {
            if al.len() != horizon || au.len() != horizon {
                return None;
            }
            if al.iter().any(|v| !v.is_finite()) || au.iter().any(|v| !v.is_finite()) {
                return None;
            }
            true
        }
        (Some(_), Some(_), _, _) => {
            return None;
        }
        _ => false,
    };

    let blended_mean: Vec<f64> = forecast_mean
        .iter()
        .zip(ar1.mean.iter())
        .map(|(&b, &a)| (1.0 - alpha) * b + alpha * a)
        .collect();
    if blended_mean.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let blended_lower = if blend_band {
        let bl = forecast_lower.unwrap();
        let al = ar1.lower_quantile.as_ref().unwrap();
        Some(
            bl.iter()
                .zip(al.iter())
                .map(|(&b, &a)| (1.0 - alpha) * b + alpha * a)
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };
    let blended_upper = if blend_band {
        let bu = forecast_upper.unwrap();
        let au = ar1.upper_quantile.as_ref().unwrap();
        Some(
            bu.iter()
                .zip(au.iter())
                .map(|(&b, &a)| (1.0 - alpha) * b + alpha * a)
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };

    if let (Some(lo), Some(hi)) = (blended_lower.as_ref(), blended_upper.as_ref()) {
        for ((m, l), u) in blended_mean.iter().zip(lo.iter()).zip(hi.iter()) {
            if !(l <= m && m <= u) {
                return None;
            }
        }
    }

    Some(BlendedForecast {
        mean: blended_mean,
        lower: blended_lower,
        upper: blended_upper,
        pred_return,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ts() -> Vec<NaiveDateTime> {
        Vec::new()
    }

    fn series_with_mean_reverting_returns(n: usize, start: f64) -> Vec<f64> {
        // r_t = -0.6 r_{t-1}, integrated.
        let mut returns = vec![1.0_f64];
        for _ in 0..n - 1 {
            let next = -0.6 * returns.last().unwrap();
            returns.push(next);
        }
        let mut values = Vec::with_capacity(n + 1);
        values.push(start);
        for r in returns {
            values.push(values.last().unwrap() + r);
        }
        values
    }

    #[test]
    fn test_skip_when_alpha_zero() {
        let mean = vec![100.0, 101.0, 102.0];
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let result = apply_revert_blend(
            &mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            100.0,
            false,
            0.0,
            0.01,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_skip_when_pred_return_below_threshold() {
        // Forecast barely moves from the anchor: pred_return ≈ 0,
        // well below a 5 % threshold. Must skip even when alpha is
        // valid.
        let mean = vec![100.0, 100.001, 100.002];
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let result = apply_revert_blend(
            &mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            100.0,
            false,
            0.5,
            0.05,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_blends_when_magnitude_exceeds_threshold() {
        // Big upward forecast. AR(1) on a mean-reverting series gives
        // a reverting prediction. The blend must move the forecast
        // toward the AR(1) value while preserving the sign of the
        // forecast (since alpha < 1).
        let mean = vec![150.0; 5];
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let anchor = *fit_values.last().unwrap();
        let result = apply_revert_blend(
            &mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            anchor,
            false,
            0.5,
            0.05,
        );
        let blended = result.expect("blend must fire for 50 % up forecast at θ = 5 %");
        for (b, m) in blended.mean.iter().zip(mean.iter()) {
            assert!(*b < *m, "blended mean {} must be pulled down from {}", b, m);
        }
    }

    #[test]
    fn test_preserves_sign_on_strong_uptrend() {
        // The convex-hull argument: blending a confident +50 % base
        // forecast with an AR(1) drift `c / (1 − φ)` of the same sign
        // (a genuine upward drift) shrinks magnitude but cannot flip
        // the sign. This locks the design invariant from
        // implementation-planning Phase 2.
        let fit_values: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let mean = vec![175.0; 5];
        let anchor = *fit_values.last().unwrap();
        let result = apply_revert_blend(
            &mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            anchor,
            false,
            0.5,
            0.05,
        );
        let blended = result.expect("blend must fire");
        for b in &blended.mean {
            assert!(
                *b > anchor,
                "blended forecast {} must remain above anchor {} (sign preserved)",
                b,
                anchor,
            );
        }
    }

    #[test]
    fn test_skip_when_base_has_band_but_ar1_lacks_one() {
        // Short fit history → AR(1) returns no band (Step 1 flat
        // fallback). Base supplied lower / upper. Blending the mean
        // while keeping the base's band would break `lower ≤ mean ≤
        // upper`, so the entire blend must be skipped.
        let mean = vec![150.0; 5];
        let lower = vec![140.0; 5];
        let upper = vec![160.0; 5];
        // Only 5 history points → below AR1_MIN_SAMPLES, fit falls
        // back to flat with no band.
        let fit_values: Vec<f64> = (0..5).map(|i| 100.0 + i as f64).collect();
        let result = apply_revert_blend(
            &mean,
            Some(&lower),
            Some(&upper),
            &fit_values,
            &empty_ts(),
            *fit_values.last().unwrap(),
            false,
            0.5,
            0.05,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_blends_mean_only_when_base_lacks_band() {
        // Base has no band, AR(1) does. The blend keeps the band
        // `None` to mirror the base's covariance shape.
        let mean = vec![150.0; 5];
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let result = apply_revert_blend(
            &mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            *fit_values.last().unwrap(),
            false,
            0.5,
            0.05,
        );
        let blended = result.expect("blend must fire");
        assert!(blended.lower.is_none());
        assert!(blended.upper.is_none());
    }

    #[test]
    fn test_log_domain_gate_uses_simple_return() {
        // log_transformed = true: pred_return = exp(Δ) − 1. A log diff
        // of 0.10 corresponds to simple +10.52 %, which exceeds a
        // 5 % threshold. With log anchor = 0 the same check at log
        // diff 0.04 is simple +4.08 %, below threshold.
        let fit_values: Vec<f64> = vec![0.0; 50]; // log domain
        let anchor = 0.0_f64;

        let big_mean = vec![0.10_f64; 5];
        let result_big = apply_revert_blend(
            &big_mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            anchor,
            true,
            0.5,
            0.05,
        );
        assert!(
            result_big.is_some(),
            "log diff 0.10 → simple +10.5 % must fire at θ = 5 %",
        );

        let small_mean = vec![0.04_f64; 5];
        let result_small = apply_revert_blend(
            &small_mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            anchor,
            true,
            0.5,
            0.05,
        );
        assert!(
            result_small.is_none(),
            "log diff 0.04 → simple +4.08 % must skip at θ = 5 %",
        );
    }

    #[test]
    fn test_rejects_non_finite_anchor() {
        let mean = vec![150.0; 5];
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let result = apply_revert_blend(
            &mean,
            None,
            None,
            &fit_values,
            &empty_ts(),
            f64::NAN,
            false,
            0.5,
            0.05,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_rejects_empty_forecast() {
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let result = apply_revert_blend(
            &[],
            None,
            None,
            &fit_values,
            &empty_ts(),
            100.0,
            false,
            0.5,
            0.05,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_blended_band_preserves_ordering() {
        // Pipe a forecast that has a band through the blend and make
        // sure the result still satisfies `lower ≤ mean ≤ upper`.
        let mean = vec![150.0; 5];
        let lower = vec![140.0; 5];
        let upper = vec![160.0; 5];
        let fit_values = series_with_mean_reverting_returns(50, 100.0);
        let result = apply_revert_blend(
            &mean,
            Some(&lower),
            Some(&upper),
            &fit_values,
            &empty_ts(),
            *fit_values.last().unwrap(),
            false,
            0.5,
            0.05,
        );
        let blended = result.expect("blend must fire");
        let blended_lower = blended.lower.expect("band must be retained");
        let blended_upper = blended.upper.expect("band must be retained");
        for ((m, l), u) in blended
            .mean
            .iter()
            .zip(blended_lower.iter())
            .zip(blended_upper.iter())
        {
            assert!(l <= m, "lower {} must be ≤ mean {}", l, m);
            assert!(m <= u, "mean {} must be ≤ upper {}", m, u);
        }
    }
}
