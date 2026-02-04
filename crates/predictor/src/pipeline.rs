use std::collections::BTreeMap;

use analyzer::TimeSeriesAnalyzer;
use chrono::{NaiveDateTime, TimeDelta};
use common::{decimals_to_f64s, f64s_to_decimals, BigDecimal, ChronosError, Result};
use normalize::normalize_time_series_data;
use selector::AdaptiveModelSelector;
use serde::{Deserialize, Serialize};
use tracing::info;
use trainer::HierarchicalTrainer;

/// Input for the prediction pipeline.
#[derive(Debug, Clone)]
pub struct PredictionInput {
    /// Time-series data as timestamp → value mapping.
    /// BTreeMap ensures data is always sorted by timestamp.
    pub data: BTreeMap<NaiveDateTime, BigDecimal>,

    /// Forecast horizon as a duration (e.g., 24 hours ahead).
    pub horizon: TimeDelta,
}

/// Full result of a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Point forecast values with timestamps.
    pub forecast_values: BTreeMap<NaiveDateTime, BigDecimal>,
    /// Lower confidence bound (10th percentile) with timestamps.
    pub lower_bound: Option<BTreeMap<NaiveDateTime, BigDecimal>>,
    /// Upper confidence bound (90th percentile) with timestamps.
    pub upper_bound: Option<BTreeMap<NaiveDateTime, BigDecimal>>,
    /// Name of model(s) used.
    pub model_name: String,
    /// Strategy selected.
    pub strategy_name: String,
    /// Total processing time in seconds.
    pub processing_time_secs: f64,
    /// Number of models trained.
    pub model_count: usize,
}

/// Calculate median sampling interval in seconds.
fn calculate_median_interval(timestamps: &[NaiveDateTime]) -> i64 {
    if timestamps.len() < 2 {
        return 1;
    }

    let mut intervals: Vec<i64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).num_seconds())
        .collect();
    let mid = intervals.len() / 2;
    (*intervals.select_nth_unstable(mid).1).max(1)
}

/// Convert TimeDelta to steps based on median sampling interval.
fn horizon_to_steps(horizon: &TimeDelta, timestamps: &[NaiveDateTime]) -> usize {
    if timestamps.len() < 2 {
        return 1;
    }

    let median_interval = calculate_median_interval(timestamps);

    // Convert horizon to steps (ceiling division, minimum 1)
    let horizon_secs = horizon.num_seconds();
    let steps = ((horizon_secs + median_interval - 1) / median_interval) as usize;
    steps.max(1)
}

/// Convert TimeDelta to steps and generate forecast timestamps.
///
/// Returns (steps, forecast_timestamps) where:
/// - steps: number of forecast steps
/// - forecast_timestamps: timestamps for each forecast point
fn horizon_to_steps_with_timestamps(
    horizon: &TimeDelta,
    timestamps: &[NaiveDateTime],
) -> (usize, Vec<NaiveDateTime>) {
    let steps = horizon_to_steps(horizon, timestamps);

    if timestamps.is_empty() {
        return (steps, Vec::new());
    }

    let last_ts = *timestamps.last().unwrap();
    let median_interval = calculate_median_interval(timestamps);

    let forecast_timestamps: Vec<NaiveDateTime> = (1..=steps)
        .map(|i| last_ts + TimeDelta::seconds(median_interval * i as i64))
        .collect();

    (steps, forecast_timestamps)
}

/// Calculate time budget based on data size.
fn calculate_time_budget(data_len: usize) -> f64 {
    // Base 60s, +10s per 100 points, max 900s
    let base = 60.0;
    let per_100_points = 10.0;
    (base + (data_len as f64 / 100.0) * per_100_points).min(900.0)
}

/// Main prediction entry point.
///
/// Pipeline: normalize → analyze → (log transform if exponential) → select strategy → train models → ensemble → (exp transform).
pub fn predict(input: &PredictionInput) -> Result<ForecastResult> {
    let start = std::time::Instant::now();

    // Validate input
    if input.data.len() < 2 {
        return Err(ChronosError::InsufficientData(
            "At least 2 data points required".into(),
        ));
    }
    if input.horizon <= TimeDelta::zero() {
        return Err(ChronosError::InvalidInput(
            "horizon must be positive".into(),
        ));
    }

    // Extract sorted data from BTreeMap
    let (timestamps, decimal_values): (Vec<_>, Vec<_>) = input
        .data
        .iter()
        .map(|(ts, val)| (*ts, val.clone()))
        .unzip();

    // Convert Decimal → f64 at the boundary (NaN/Inf cannot exist in Decimal)
    let f64_values = decimals_to_f64s(&decimal_values)?;

    // Auto-calculate time budget based on data size
    let time_budget = calculate_time_budget(input.data.len());

    info!(
        data_points = input.data.len(),
        horizon = %input.horizon,
        time_budget = format!("{:.0}s", time_budget),
        "Starting prediction pipeline"
    );

    // Step 1: Normalize
    let (norm_timestamps, norm_values) = normalize_time_series_data(&timestamps, &f64_values)?;

    // Step 2: Analyze characteristics (run once, share with selector and trainer)
    let analyzer = TimeSeriesAnalyzer::new();
    let characteristics = analyzer.analyze(&norm_values, &norm_timestamps);
    let season_period = characteristics.seasonality.period;
    let is_exponential = characteristics.trend.is_exponential;

    // Step 2.5: Apply log transform if exponential trend detected
    let (train_values, log_transformed) = if is_exponential {
        info!("Exponential trend detected, applying log transform");
        let log_vals: Vec<f64> = norm_values.iter().map(|v| v.ln()).collect();
        (log_vals, true)
    } else {
        (norm_values.clone(), false)
    };

    // Step 3: Select strategy from pre-computed characteristics
    let selector = AdaptiveModelSelector::default();
    let strategy = selector.select_strategy_from_characteristics(
        &characteristics,
        train_values.len(),
        time_budget as u64,
    );

    // Convert horizon from TimeDelta to steps with timestamps
    let (horizon_steps, forecast_timestamps) =
        horizon_to_steps_with_timestamps(&input.horizon, &norm_timestamps);

    // Step 4: Hierarchical training + ensemble (with detected season period)
    let mut trainer = HierarchicalTrainer::default();
    let hints = trainer::TrainingHints {
        season_period,
        volatility: Some(characteristics.volatility),
    };
    let (forecast, metadata) = trainer.train_hierarchically(
        &train_values,
        &norm_timestamps,
        &strategy,
        time_budget,
        horizon_steps,
        hints,
    )?;

    // Step 5: Inverse transform if log was applied
    let final_mean: Vec<f64> = if log_transformed {
        forecast.mean.iter().map(|v| v.exp()).collect()
    } else {
        forecast.mean
    };

    let final_lower: Option<Vec<f64>> = if log_transformed {
        forecast
            .lower_quantile
            .map(|v| v.iter().map(|x| x.exp()).collect())
    } else {
        forecast.lower_quantile
    };

    let final_upper: Option<Vec<f64>> = if log_transformed {
        forecast
            .upper_quantile
            .map(|v| v.iter().map(|x| x.exp()).collect())
    } else {
        forecast.upper_quantile
    };

    let processing_time = start.elapsed().as_secs_f64();

    info!(
        strategy = %strategy.strategy_name,
        models = metadata.model_count,
        log_transformed = log_transformed,
        time = format!("{:.2}s", processing_time),
        "Prediction pipeline complete"
    );

    // Convert f64 → Decimal and build BTreeMap with timestamps
    let forecast_decimals = f64s_to_decimals(&final_mean)?;
    let forecast_values: BTreeMap<NaiveDateTime, BigDecimal> = forecast_timestamps
        .iter()
        .zip(forecast_decimals)
        .map(|(ts, val)| (*ts, val))
        .collect();

    let lower_bound = final_lower
        .map(|v| {
            f64s_to_decimals(&v).map(|decimals| {
                forecast_timestamps
                    .iter()
                    .zip(decimals.into_iter())
                    .map(|(ts, val)| (*ts, val))
                    .collect()
            })
        })
        .transpose()?;

    let upper_bound = final_upper
        .map(|v| {
            f64s_to_decimals(&v).map(|decimals| {
                forecast_timestamps
                    .iter()
                    .zip(decimals.into_iter())
                    .map(|(ts, val)| (*ts, val))
                    .collect()
            })
        })
        .transpose()?;

    Ok(ForecastResult {
        forecast_values,
        lower_bound,
        upper_bound,
        model_name: forecast.model_name,
        strategy_name: strategy.strategy_name,
        processing_time_secs: processing_time,
        model_count: metadata.model_count,
    })
}

#[cfg(test)]
mod tests;
