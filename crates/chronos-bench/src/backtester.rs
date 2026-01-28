use chronos_core::{ForecastModel, ForecastOutput};
use chronos_models::{EtsModel, NptsModel, SeasonalNaiveModel, ThetaModel};
use chronos_selector::AdaptiveModelSelector;
use chronos_trainer::HierarchicalTrainer;
use tracing::{debug, warn};

use crate::data_generator::TimeSeriesFixture;
use crate::metrics::{self, MetricSet};

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
    let model_configs: Vec<(&str, Box<dyn FnOnce() -> Box<dyn ForecastModel>>)> = vec![
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
    let selector = AdaptiveModelSelector::default();
    let strategy = selector.select_optimal_strategy(
        &fixture.train_values,
        &fixture.train_timestamps,
        fixture.horizon,
        60,
    );

    let mut trainer = HierarchicalTrainer::default();
    match trainer.train_hierarchically(
        &fixture.train_values,
        &fixture.train_timestamps,
        &strategy,
        60.0,
        fixture.horizon,
        fixture.expected_characteristics.seasonal_period,
    ) {
        Ok((forecast, _metadata)) => Some(forecast),
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
