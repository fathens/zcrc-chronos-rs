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
    /// Point forecast values.
    pub forecast_values: Vec<BigDecimal>,
    /// Lower confidence bound (10th percentile).
    pub lower_bound: Option<Vec<BigDecimal>>,
    /// Upper confidence bound (90th percentile).
    pub upper_bound: Option<Vec<BigDecimal>>,
    /// Name of model(s) used.
    pub model_name: String,
    /// Strategy selected.
    pub strategy_name: String,
    /// Total processing time in seconds.
    pub processing_time_secs: f64,
    /// Number of models trained.
    pub model_count: usize,
}

/// Convert TimeDelta to steps based on median sampling interval.
fn horizon_to_steps(horizon: &TimeDelta, timestamps: &[NaiveDateTime]) -> usize {
    if timestamps.len() < 2 {
        return 1;
    }

    // Calculate median sampling interval
    let mut intervals: Vec<i64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).num_seconds())
        .collect();
    intervals.sort();
    let median_interval = intervals[intervals.len() / 2];

    // Convert horizon to steps (minimum 1)
    let steps = (horizon.num_seconds() / median_interval.max(1)) as usize;
    steps.max(1)
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

    // Convert horizon from TimeDelta to steps
    let horizon_steps = horizon_to_steps(&input.horizon, &norm_timestamps);

    // Step 4: Hierarchical training + ensemble (with detected season period)
    let mut trainer = HierarchicalTrainer::default();
    let (forecast, metadata) = trainer.train_hierarchically(
        &train_values,
        &norm_timestamps,
        &strategy,
        time_budget,
        horizon_steps,
        season_period,
    )?;

    // Step 5: Inverse transform if log was applied
    let final_mean = if log_transformed {
        forecast.mean.iter().map(|v| v.exp()).collect()
    } else {
        forecast.mean
    };

    let final_lower = if log_transformed {
        forecast
            .lower_quantile
            .map(|v| v.iter().map(|x| x.exp()).collect())
    } else {
        forecast.lower_quantile
    };

    let final_upper = if log_transformed {
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

    // Convert f64 → Decimal at the boundary
    Ok(ForecastResult {
        forecast_values: f64s_to_decimals(&final_mean)?,
        lower_bound: final_lower.map(|v| f64s_to_decimals(&v)).transpose()?,
        upper_bound: final_upper.map(|v| f64s_to_decimals(&v)).transpose()?,
        model_name: forecast.model_name,
        strategy_name: strategy.strategy_name,
        processing_time_secs: processing_time,
        model_count: metadata.model_count,
    })
}

#[cfg(test)]
mod tests;
