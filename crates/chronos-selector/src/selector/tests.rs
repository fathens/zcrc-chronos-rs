use super::*;
use chrono::NaiveDate;

fn uniform_timestamps(n: usize, interval_secs: i64) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::seconds(interval_secs * i as i64))
        .collect()
}

#[test]
fn test_default_balanced_strategy() {
    let selector = AdaptiveModelSelector::default();
    // Flat data with enough points → balanced
    let vals = vec![100.0; 200];
    let ts = uniform_timestamps(200, 3600);
    let strategy = selector.select_optimal_strategy(&vals, &ts, 24, 900);
    // Should not crash; strategy name should be populated
    assert!(!strategy.strategy_name.is_empty());
    assert!(!strategy.priority_models.is_empty());
}

#[test]
fn test_small_dataset_strategy() {
    let selector = AdaptiveModelSelector::new(Some(100), Some(1000));
    let vals = vec![50.0; 30]; // < 100 points → small_dataset base
    let ts = uniform_timestamps(30, 3600);
    let strategy = selector.select_optimal_strategy(&vals, &ts, 10, 900);
    // With flat data, no strong characteristics → should use base strategy
    assert!(
        strategy.strategy_name.contains("small_dataset")
            || strategy.strategy_name == "balanced"
    );
}

#[test]
fn test_strong_trend_strategy() {
    let selector = AdaptiveModelSelector::default();
    // Strong linear uptrend
    let vals: Vec<f64> = (0..200).map(|i| i as f64 * 10.0 + 100.0).collect();
    let ts = uniform_timestamps(200, 3600);
    let strategy = selector.select_optimal_strategy(&vals, &ts, 24, 900);
    assert!(
        strategy.strategy_name.contains("trend"),
        "Expected trend strategy, got: {}",
        strategy.strategy_name
    );
}

#[test]
fn test_seasonal_strategy() {
    let selector = AdaptiveModelSelector::default();
    // Strong seasonal signal with large DC offset to keep volatility low
    let n = 200;
    let vals: Vec<f64> = (0..n)
        .map(|i| 1000.0 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() * 50.0)
        .collect();
    let ts = uniform_timestamps(n, 3600);
    let strategy = selector.select_optimal_strategy(&vals, &ts, 24, 900);
    assert!(
        strategy.strategy_name.contains("seasonal"),
        "Expected seasonal strategy, got: {}",
        strategy.strategy_name
    );
}

#[test]
fn test_short_time_budget_adjustment() {
    let selector = AdaptiveModelSelector::default();
    let vals = vec![100.0; 200];
    let ts = uniform_timestamps(200, 3600);
    let strategy = selector.select_optimal_strategy(&vals, &ts, 24, 120); // 2 min
    assert!(strategy.strategy_name.contains("fast"));
    assert_eq!(strategy.preset, "fast_training");
}

#[test]
fn test_long_time_budget_adjustment() {
    let selector = AdaptiveModelSelector::default();
    let vals = vec![100.0; 200];
    let ts = uniform_timestamps(200, 3600);
    let strategy = selector.select_optimal_strategy(&vals, &ts, 24, 3600); // 60 min
    assert!(strategy.strategy_name.contains("extended"));
    assert_eq!(strategy.preset, "high_quality");
}

#[test]
fn test_hierarchical_model_groups() {
    let strategy = ModelSelectionStrategy {
        strategy_name: "test".into(),
        priority_models: vec![
            "SeasonalNaive".into(),
            "AutoETS".into(),
            "RecursiveTabular".into(),
            "NPTS".into(),
            "DeepAR".into(),
        ],
        excluded_models: vec![],
        time_allocation: TimeAllocation {
            fast: 0.2,
            medium: 0.5,
            advanced: 0.3,
        },
        preset: "medium_quality".into(),
    };

    let groups = AdaptiveModelSelector::get_hierarchical_model_groups(&strategy);
    assert!(groups.contains_key("fast"));
    assert!(groups.contains_key("medium"));
    assert!(groups.contains_key("advanced"));
    assert!(groups["fast"].contains(&"SeasonalNaive".to_string()));
    assert!(groups["medium"].contains(&"NPTS".to_string()));
    assert!(groups["advanced"].contains(&"DeepAR".to_string()));
}

#[test]
fn test_estimate_training_time() {
    assert_eq!(
        AdaptiveModelSelector::estimate_model_training_time("SeasonalNaive", 50),
        1.0
    );
    assert_eq!(
        AdaptiveModelSelector::estimate_model_training_time("SeasonalNaive", 200),
        1.5
    );
    assert_eq!(
        AdaptiveModelSelector::estimate_model_training_time("DeepAR", 1000),
        120.0
    );
}

#[test]
fn test_irregular_strategy() {
    let selector = AdaptiveModelSelector::default();
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    // Highly irregular timestamps
    let ts: Vec<NaiveDateTime> = (0..200)
        .map(|i| {
            let gap = if i % 3 == 0 { 60 } else { 7200 };
            base + chrono::Duration::seconds(gap * i as i64)
        })
        .collect();
    let vals = vec![100.0; 200];
    let strategy = selector.select_optimal_strategy(&vals, &ts, 24, 900);
    assert!(
        strategy.strategy_name.contains("irregular"),
        "Expected irregular strategy, got: {}",
        strategy.strategy_name
    );
}

/// All strategies that may be used for seasonal data should include MSTL.
/// This is a regression test: small_dataset previously lacked MSTL, causing
/// poor ensemble performance on multi_seasonal_100.
#[test]
fn test_all_strategies_include_key_fast_models() {
    let strategies = super::initialize_strategies();

    // Strategies that should include MSTL (may encounter seasonal data)
    let strategies_needing_mstl = [
        "strong_seasonal",
        "small_dataset",
        "balanced",
    ];

    for name in strategies_needing_mstl {
        let strategy = strategies.get(name).expect(&format!("Strategy {} not found", name));
        assert!(
            strategy.priority_models.iter().any(|m| m == "MSTL"),
            "Strategy '{}' should include MSTL in priority_models",
            name
        );
    }

    // All fast-category models should be available for short time budgets
    let fast_models = ["SeasonalNaive", "AutoETS", "ETS", "Theta", "MSTL"];
    for (name, strategy) in &strategies {
        // After short-time adjustment, fast models should remain available
        let adjusted = super::adjust_for_short_time(strategy);
        for model in fast_models {
            // Model should either be in priority or not in excluded
            let in_priority = adjusted.priority_models.iter().any(|m| m == model);
            let in_excluded = adjusted.excluded_models.iter().any(|m| m == model);
            // If model is in the original strategy's priority, it should remain after adjustment
            if strategy.priority_models.iter().any(|m| m == model) {
                assert!(
                    in_priority && !in_excluded,
                    "Strategy '{}': fast model '{}' should not be excluded after short-time adjustment",
                    name,
                    model
                );
            }
        }
    }
}
