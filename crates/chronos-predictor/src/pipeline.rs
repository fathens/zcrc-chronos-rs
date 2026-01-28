use chrono::NaiveDateTime;
use chronos_core::{ChronosError, Result};
use chronos_normalize::normalize_time_series_data;
use chronos_selector::AdaptiveModelSelector;
use chronos_trainer::HierarchicalTrainer;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Input for the prediction pipeline.
#[derive(Debug, Clone)]
pub struct PredictionInput {
    pub timestamps: Vec<NaiveDateTime>,
    pub values: Vec<f64>,
    pub horizon: usize,
    pub time_budget_secs: Option<f64>,
}

/// Full result of a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Point forecast values.
    pub forecast_values: Vec<f64>,
    /// Lower confidence bound (10th percentile).
    pub lower_bound: Option<Vec<f64>>,
    /// Upper confidence bound (90th percentile).
    pub upper_bound: Option<Vec<f64>>,
    /// Name of model(s) used.
    pub model_name: String,
    /// Strategy selected.
    pub strategy_name: String,
    /// Total processing time in seconds.
    pub processing_time_secs: f64,
    /// Number of models trained.
    pub model_count: usize,
}

/// Main prediction entry point.
///
/// Pipeline: normalize → analyze → select strategy → train models → ensemble.
pub fn predict(input: &PredictionInput) -> Result<ForecastResult> {
    let start = std::time::Instant::now();

    // Validate input
    if input.timestamps.len() != input.values.len() {
        return Err(ChronosError::InvalidInput(
            "timestamps and values must have the same length".into(),
        ));
    }
    if input.values.len() < 2 {
        return Err(ChronosError::InsufficientData(
            "At least 2 data points required".into(),
        ));
    }
    if input.horizon == 0 {
        return Err(ChronosError::InvalidInput(
            "horizon must be positive".into(),
        ));
    }

    // Check for NaN / Inf
    for (i, v) in input.values.iter().enumerate() {
        if v.is_nan() || v.is_infinite() {
            return Err(ChronosError::InvalidInput(format!(
                "Invalid value at index {i}: {v}"
            )));
        }
    }

    let time_budget = input.time_budget_secs.unwrap_or(900.0);

    info!(
        data_points = input.values.len(),
        horizon = input.horizon,
        time_budget = format!("{:.0}s", time_budget),
        "Starting prediction pipeline"
    );

    // Step 1: Normalize
    let (norm_timestamps, norm_values) =
        normalize_time_series_data(&input.timestamps, &input.values)?;

    // Step 2: Select strategy
    let selector = AdaptiveModelSelector::default();
    let strategy =
        selector.select_optimal_strategy(&norm_values, &norm_timestamps, input.horizon, time_budget as u64);

    // Step 3: Hierarchical training + ensemble
    let mut trainer = HierarchicalTrainer::default();
    let (forecast, metadata) = trainer.train_hierarchically(
        &norm_values,
        &norm_timestamps,
        &strategy,
        time_budget,
        input.horizon,
    )?;

    let processing_time = start.elapsed().as_secs_f64();

    info!(
        strategy = %strategy.strategy_name,
        models = metadata.model_count,
        time = format!("{:.2}s", processing_time),
        "Prediction pipeline complete"
    );

    Ok(ForecastResult {
        forecast_values: forecast.mean,
        lower_bound: forecast.lower_quantile,
        upper_bound: forecast.upper_quantile,
        model_name: forecast.model_name,
        strategy_name: strategy.strategy_name,
        processing_time_secs: processing_time,
        model_count: metadata.model_count,
    })
}

#[cfg(test)]
mod tests;
