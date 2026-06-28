use augurs::prelude::*;
use chrono::NaiveDateTime;
use common::{ChronosError, ForecastModel, ForecastOutput, ModelCategory, Result};
use tracing::debug;

use crate::hw;

/// Non-seasonal augurs AutoETS specification used by [`EtsModel`],
/// [`crate::mstl::MstlEtsModel`], and the theta-2 line of
/// [`crate::theta::ThetaModel`].
///
/// `"ZAN"` means: error component selected by AICc, **trend forced to
/// additive (damped or undamped, AICc picks)**, no seasonal component.
///
/// The previous spec `"ZZN"` let AICc also pick "no trend" (ETS(A,N,N)).
/// On low-SNR series — common in crypto-token price data — the AICc
/// penalty for the extra trend parameter often outweighed the
/// trend-fitting improvement, and the predictor returned a perfectly
/// flat forecast for the entire horizon. The
/// `real_data_over_damping` diagnostic showed this affecting >50 % of
/// the existing real fixtures. Forcing `Additive` removes the
/// "no-trend" branch from the AICc search; the resulting damped/
/// undamped pair still allows the slope to shrink toward zero when the
/// data genuinely lacks a trend, but the forecast is no longer
/// identically constant.
pub(crate) const ETS_SPEC: &str = "ZAN";

/// ETS model wrapper around augurs AutoETS.
///
/// When `season_length > 1`, uses a custom Holt-Winters implementation
/// that supports additive and multiplicative seasonality.
/// Otherwise falls back to augurs AutoETS with "ZAN" (non-seasonal,
/// additive trend) spec — see `ETS_SPEC` below for the rationale.
pub struct EtsModel {
    season_length: Option<usize>,
}

impl EtsModel {
    pub fn new(season_length: Option<usize>) -> Self {
        Self { season_length }
    }
}

impl ForecastModel for EtsModel {
    fn name(&self) -> &str {
        "ETS"
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
        if values.len() < 3 {
            return Err(ChronosError::InsufficientData(
                "ETS requires at least 3 data points".into(),
            ));
        }

        // Seasonal ETS: delegate to Holt-Winters implementation
        // Falls back to non-seasonal if insufficient data for 2 full cycles
        if let Some(season_len) = self.season_length.filter(|&s| s > 1) {
            if values.len() >= 2 * season_len {
                debug!(
                    season_length = season_len,
                    horizon = horizon,
                    data_length = values.len(),
                    "ETS fitting (Holt-Winters seasonal)"
                );

                let result = hw::hw_fit_predict(values, season_len, horizon)?;
                return Ok(ForecastOutput {
                    mean: result.mean,
                    lower_quantile: Some(result.lower),
                    upper_quantile: Some(result.upper),
                    model_name: "ETS".into(),
                });
            }
            // Insufficient data for seasonal model - fall through to non-seasonal
            debug!(
                season_length = season_len,
                required = 2 * season_len,
                actual = values.len(),
                "Insufficient data for seasonal ETS, falling back to non-seasonal"
            );
        }

        // Non-seasonal: use augurs AutoETS with `ETS_SPEC`.
        let spec = ETS_SPEC;
        let season_len = 1;

        debug!(
            requested_season_length = ?self.season_length,
            season_length = season_len,
            spec = spec,
            horizon = horizon,
            data_length = values.len(),
            "ETS fitting (non-seasonal)"
        );

        let auto = augurs::ets::AutoETS::new(season_len, spec)
            .map_err(|e| ChronosError::ModelError(format!("ETS init: {e}")))?;

        let fitted = auto
            .fit(values)
            .map_err(|e| ChronosError::ModelError(format!("ETS fit: {e}")))?;

        let forecast = fitted
            .predict(horizon, 0.80)
            .map_err(|e| ChronosError::ModelError(format!("ETS predict: {e}")))?;

        let lower = forecast.intervals.as_ref().map(|iv| iv.lower.clone());
        let upper = forecast.intervals.as_ref().map(|iv| iv.upper.clone());

        Ok(ForecastOutput {
            mean: forecast.point,
            lower_quantile: lower,
            upper_quantile: upper,
            model_name: "ETS".into(),
        })
    }
}

#[cfg(test)]
mod tests;
