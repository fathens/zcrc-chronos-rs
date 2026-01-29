use common::{decimals_to_f64s, f64s_to_decimals, BigDecimal, ForecastModel, ForecastOutput};
use models::{EtsModel, MstlEtsModel, NptsModel, SeasonalNaiveModel, ThetaModel};
use predictor::{predict, PredictionInput};
use tracing::{debug, warn};

use crate::data_generator::TimeSeriesFixture;
use crate::metrics::{self, MetricSet};

type ModelFactory<'a> = Box<dyn FnOnce() -> Box<dyn ForecastModel> + 'a>;

/// Result of running a single model on a single fixture.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub fixture_name: String,
    pub model_name: String,
    pub metrics: MetricSet,
    pub forecast: Vec<f64>,
    pub actual: Vec<f64>,
}

/// Run all individual models + the full ensemble pipeline on a single fixture.
pub fn run_backtest(fixture: &TimeSeriesFixture) -> Vec<BacktestResult> {
    let season = fixture
        .expected_characteristics
        .seasonal_period
        .unwrap_or(1);

    let mut results = Vec::new();

    // Individual models
    let model_configs: Vec<(&str, ModelFactory<'_>)> = vec![
        (
            "SeasonalNaive",
            Box::new(move || {
                Box::new(SeasonalNaiveModel::new(
                    fixture.expected_characteristics.seasonal_period,
                )) as Box<dyn ForecastModel>
            }),
        ),
        (
            "ETS",
            Box::new(move || {
                Box::new(EtsModel::new(
                    fixture.expected_characteristics.seasonal_period,
                )) as Box<dyn ForecastModel>
            }),
        ),
        (
            "Theta",
            Box::new(|| Box::new(ThetaModel::new()) as Box<dyn ForecastModel>),
        ),
        (
            "MSTL",
            Box::new(move || {
                let periods = fixture
                    .expected_characteristics
                    .seasonal_period
                    .map(|p| vec![p]);
                Box::new(MstlEtsModel::new(periods)) as Box<dyn ForecastModel>
            }),
        ),
        (
            "NPTS",
            Box::new(|| Box::new(NptsModel::default()) as Box<dyn ForecastModel>),
        ),
    ];

    for (name, make_model) in model_configs {
        match run_single_model(make_model(), fixture) {
            Some(forecast_output) => {
                let forecast = &forecast_output.mean;
                let ms = metrics::compute_metrics(
                    forecast,
                    &fixture.test_values,
                    &fixture.train_values,
                    season,
                );
                results.push(BacktestResult {
                    fixture_name: fixture.name.clone(),
                    model_name: name.to_string(),
                    metrics: ms,
                    forecast: forecast.clone(),
                    actual: fixture.test_values.clone(),
                });
            }
            None => {
                warn!(model = name, fixture = %fixture.name, "Model failed");
            }
        }
    }

    // Full pipeline (ensemble)
    if let Some(ensemble_result) = run_ensemble_pipeline(fixture) {
        let ms = metrics::compute_metrics(
            &ensemble_result.mean,
            &fixture.test_values,
            &fixture.train_values,
            season,
        );
        results.push(BacktestResult {
            fixture_name: fixture.name.clone(),
            model_name: ensemble_result.model_name.clone(),
            metrics: ms,
            forecast: ensemble_result.mean.clone(),
            actual: fixture.test_values.clone(),
        });
    }

    results
}

fn run_single_model(
    mut model: Box<dyn ForecastModel>,
    fixture: &TimeSeriesFixture,
) -> Option<ForecastOutput> {
    match model.fit_predict(
        &fixture.train_values,
        &fixture.train_timestamps,
        fixture.horizon,
    ) {
        Ok(output) => Some(output),
        Err(e) => {
            debug!(error = %e, "Model failed on fixture");
            None
        }
    }
}

fn run_ensemble_pipeline(fixture: &TimeSeriesFixture) -> Option<ForecastOutput> {
    // Convert f64 to BigDecimal for predict() pipeline
    let values: Vec<BigDecimal> = match f64s_to_decimals(&fixture.train_values) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, fixture = %fixture.name, "Failed to convert values");
            return None;
        }
    };

    let input = PredictionInput {
        timestamps: fixture.train_timestamps.clone(),
        values,
        horizon: fixture.horizon,
        time_budget_secs: Some(60.0),
    };

    match predict(&input) {
        Ok(result) => {
            // Convert BigDecimal back to f64
            let mean = match decimals_to_f64s(&result.forecast_values) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, fixture = %fixture.name, "Failed to convert forecast");
                    return None;
                }
            };

            let lower_quantile = result.lower_bound.and_then(|lb| decimals_to_f64s(&lb).ok());
            let upper_quantile = result.upper_bound.and_then(|ub| decimals_to_f64s(&ub).ok());

            Some(ForecastOutput {
                mean,
                lower_quantile,
                upper_quantile,
                model_name: result.model_name,
            })
        }
        Err(e) => {
            warn!(error = %e, fixture = %fixture.name, "Ensemble pipeline failed");
            None
        }
    }
}

/// Run backtests on all provided fixtures.
pub fn run_all_backtests(fixtures: &[TimeSeriesFixture]) -> Vec<BacktestResult> {
    fixtures.iter().flat_map(run_backtest).collect()
}
