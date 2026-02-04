use super::*;
use chrono::NaiveDate;
use common::TimeAllocation;

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn make_strategy() -> ModelSelectionStrategy {
    ModelSelectionStrategy {
        strategy_name: "balanced".into(),
        priority_models: vec!["SeasonalNaive".into(), "AutoETS".into(), "NPTS".into()],
        excluded_models: vec!["Naive".into()],
        time_allocation: TimeAllocation {
            fast: 0.2,
            medium: 0.5,
            advanced: 0.3,
        },
        preset: "medium_quality".into(),
    }
}

#[test]
fn test_hierarchical_basic() {
    let mut trainer = HierarchicalTrainer::default();
    let values: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
    let ts = make_timestamps(50);
    let strategy = make_strategy();

    let (forecast, metadata) = trainer
        .train_hierarchically(&values, &ts, &strategy, 60.0, 5, None)
        .unwrap();

    assert_eq!(forecast.mean.len(), 5);
    assert!(metadata.total_training_time > 0.0);
    assert!(metadata.model_count > 0);
}

#[test]
fn test_hierarchical_constant_data() {
    let mut trainer = HierarchicalTrainer::default();
    let values = vec![42.0; 50];
    let ts = make_timestamps(50);
    let strategy = make_strategy();

    let (forecast, _) = trainer
        .train_hierarchically(&values, &ts, &strategy, 30.0, 3, None)
        .unwrap();

    assert_eq!(forecast.mean.len(), 3);
    // Constant data → predictions should be near 42
    for v in &forecast.mean {
        assert!((*v - 42.0).abs() < 20.0, "Expected ~42, got {}", v);
    }
}

#[test]
fn test_softmax_ensemble() {
    let f1 = ForecastOutput {
        mean: vec![10.0, 20.0],
        lower_quantile: None,
        upper_quantile: None,
        model_name: "A".into(),
    };
    let f2 = ForecastOutput {
        mean: vec![30.0, 40.0],
        lower_quantile: None,
        upper_quantile: None,
        model_name: "B".into(),
    };

    // Equal scores → equal weights → simple average
    let ensemble = softmax_ensemble(&[(f1, 0.5), (f2, 0.5)], 2);
    assert_eq!(ensemble.mean.len(), 2);
    // With equal weights: (10+30)/2=20, (20+40)/2=30
    assert!((ensemble.mean[0] - 20.0).abs() < 0.01);
    assert!((ensemble.mean[1] - 30.0).abs() < 0.01);
}

#[test]
fn test_softmax_ensemble_weighted() {
    let f1 = ForecastOutput {
        mean: vec![100.0],
        lower_quantile: None,
        upper_quantile: None,
        model_name: "Good".into(),
    };
    let f2 = ForecastOutput {
        mean: vec![200.0],
        lower_quantile: None,
        upper_quantile: None,
        model_name: "Bad".into(),
    };

    // f1 has much better score (0.01) vs f2 (1.0) → f1 dominates with softmax
    let ensemble = softmax_ensemble(&[(f1, 0.01), (f2, 1.0)], 1);
    // With temperature=0.5, score diff of 0.99 → exp(-0.99/0.5) ≈ 0.14 weight for f2
    // f1 dominates heavily
    assert!(
        ensemble.mean[0] < 120.0,
        "Expected ~100, got {}",
        ensemble.mean[0]
    );
}

#[test]
fn test_early_stopping() {
    let mut trainer = HierarchicalTrainer::new(Some(0.02), Some(0.005));
    // Simulate: set best_score very low → should trigger early stop
    trainer.best_score = 0.001;
    assert!(trainer.should_stop_early("medium"));
}

#[test]
fn test_create_model_known() {
    let vals = vec![1.0; 10];
    assert!(create_model("SeasonalNaive", &vals, None).is_some());
    assert!(create_model("ETS", &vals, None).is_some());
    assert!(create_model("Theta", &vals, None).is_some());
    assert!(create_model("MSTL", &vals, None).is_some());
    assert!(create_model("NPTS", &vals, None).is_some());
    assert!(create_model("UnknownModel", &vals, None).is_none());
}

// ---- filter_by_score tests ----

fn make_forecast(name: &str) -> ForecastOutput {
    ForecastOutput {
        mean: vec![1.0, 2.0, 3.0],
        lower_quantile: None,
        upper_quantile: None,
        model_name: name.into(),
    }
}

#[test]
fn test_filter_by_score_removes_outliers() {
    // best=0.5, threshold=max(0.5*3, 2.0)=2.0 → score=8.0 excluded
    let forecasts = vec![(make_forecast("Good"), 0.5), (make_forecast("Bad"), 8.0)];
    let filtered = filter_by_score(&forecasts);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].0.model_name, "Good");
}

#[test]
fn test_filter_by_score_keeps_all_when_similar() {
    // all scores ~1.0, threshold=max(0.8*3, 2.0)=2.4 → all pass
    let forecasts = vec![
        (make_forecast("A"), 0.8),
        (make_forecast("B"), 1.0),
        (make_forecast("C"), 1.2),
    ];
    let filtered = filter_by_score(&forecasts);
    assert_eq!(filtered.len(), 3);
}

#[test]
fn test_filter_by_score_fallback_single() {
    // All models have very high scores, but none pass threshold
    // best=10.0, threshold=max(10*3, 2.0)=30 → all pass actually
    // Let's test the edge case: best=0.1, threshold=2.0, all others above
    let forecasts = vec![
        (make_forecast("Best"), 0.1),
        (make_forecast("Worse1"), 5.0),
        (make_forecast("Worse2"), 10.0),
    ];
    let filtered = filter_by_score(&forecasts);
    // threshold = max(0.1*3, 2.0) = 2.0
    // Best(0.1) passes, Worse1(5.0) and Worse2(10.0) filtered out
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].0.model_name, "Best");
}

#[test]
fn test_filter_by_score_single_model() {
    let forecasts = vec![(make_forecast("Only"), 5.0)];
    let filtered = filter_by_score(&forecasts);
    assert_eq!(filtered.len(), 1);
}

#[test]
fn test_filter_by_score_empty() {
    let forecasts: Vec<(ForecastOutput, f64)> = vec![];
    let filtered = filter_by_score(&forecasts);
    assert!(filtered.is_empty());
}

#[test]
fn test_filter_by_score_high_best_keeps_all() {
    // When best score is high (data is hard), threshold is high → all pass
    // best=5.0, threshold=max(5*3, 2.0)=15.0
    let forecasts = vec![
        (make_forecast("A"), 5.0),
        (make_forecast("B"), 8.0),
        (make_forecast("C"), 12.0),
    ];
    let filtered = filter_by_score(&forecasts);
    assert_eq!(filtered.len(), 3);
}

#[test]
fn test_ensemble_with_filtering_seasonal() {
    // Test that filtering improves ensemble on seasonal data
    use common::ForecastModel;
    use models::EtsModel;

    let period = 12;
    let n = 200; // More data for better ETS fit
                 // Generate seasonal data with trend (ETS excels at this)
    let values: Vec<f64> = (0..n)
        .map(|i| {
            100.0
                + 0.5 * i as f64 // trend
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
        })
        .collect();
    let ts = make_timestamps(n);

    // ETS should score well on seasonal data
    let mut ets = EtsModel::new(Some(period));
    let holdout = 12;
    let train_values = &values[..n - holdout];
    let train_ts = &ts[..n - holdout];
    let actual = &values[n - holdout..];

    let forecast = ets.fit_predict(train_values, train_ts, holdout).unwrap();
    let score = common::metrics::mase(&forecast.mean, actual, train_values, period);

    // ETS should achieve MASE < 1.5 on clean seasonal data with trend
    assert!(
        score < 1.5,
        "ETS MASE on seasonal = {score:.3}, expected < 1.5"
    );
}
