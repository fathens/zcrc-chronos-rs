use super::*;
use chrono::NaiveDate;
use chronos_core::TimeAllocation;

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
        priority_models: vec![
            "SeasonalNaive".into(),
            "AutoETS".into(),
            "NPTS".into(),
        ],
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
        assert!(
            (*v - 42.0).abs() < 20.0,
            "Expected ~42, got {}",
            v
        );
    }
}

#[test]
fn test_inverse_mae_ensemble() {
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
    let ensemble = inverse_mae_ensemble(&[(f1, 0.5), (f2, 0.5)], 2);
    assert_eq!(ensemble.mean.len(), 2);
    // With equal weights: (10+30)/2=20, (20+40)/2=30
    assert!((ensemble.mean[0] - 20.0).abs() < 0.01);
    assert!((ensemble.mean[1] - 30.0).abs() < 0.01);
}

#[test]
fn test_inverse_mae_ensemble_weighted() {
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

    // f1 has much better score (0.01) vs f2 (1.0) → f1 dominates
    let ensemble = inverse_mae_ensemble(&[(f1, 0.01), (f2, 1.0)], 1);
    // f1 weight: 1/0.01 = 100, f2 weight: 1/1.0 = 1 → f1 dominates
    assert!(ensemble.mean[0] < 110.0, "Expected ~100, got {}", ensemble.mean[0]);
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
    assert!(create_model("NPTS", &vals, None).is_some());
    assert!(create_model("UnknownModel", &vals, None).is_none());
}
