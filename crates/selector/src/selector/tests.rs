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
        strategy.strategy_name.contains("small_dataset") || strategy.strategy_name == "balanced"
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

// ============================================================================
// Threshold and Scoring Boundary Tests
// ============================================================================

/// Test seasonality threshold boundary conditions.
/// The condition is: strength == "strong" AND score > 0.3
#[test]
fn test_seasonality_threshold_boundary() {
    use common::TimeSeriesCharacteristics;

    let selector = AdaptiveModelSelector::default();

    // score = 0.29, strength = "strong" → condition NOT met (score <= 0.3)
    let mut chars = TimeSeriesCharacteristics::default();
    chars.seasonality.strength = "strong".into();
    chars.seasonality.score = 0.29;
    chars.density.regular = true;
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        !strategy.strategy_name.contains("seasonal"),
        "score=0.29 should NOT trigger seasonal: got {}",
        strategy.strategy_name
    );

    // score = 0.31, strength = "strong" → condition met
    chars.seasonality.score = 0.31;
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("seasonal"),
        "score=0.31 should trigger seasonal: got {}",
        strategy.strategy_name
    );
}

/// Test trend threshold boundary conditions.
/// The condition is: strength == "strong" AND r_squared > 0.7
#[test]
fn test_trend_threshold_boundary() {
    use common::TimeSeriesCharacteristics;

    let selector = AdaptiveModelSelector::default();

    // r² = 0.69, strength = "strong" → condition NOT met (r² <= 0.7)
    let mut chars = TimeSeriesCharacteristics::default();
    chars.trend.strength = "strong".into();
    chars.trend.r_squared = 0.69;
    chars.density.regular = true;
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        !strategy.strategy_name.contains("trend"),
        "r²=0.69 should NOT trigger trend: got {}",
        strategy.strategy_name
    );

    // r² = 0.71, strength = "strong" → condition met
    chars.trend.r_squared = 0.71;
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("trend"),
        "r²=0.71 should trigger trend: got {}",
        strategy.strategy_name
    );
}

/// Test volatility threshold boundary conditions.
/// The condition is: volatility > 0.15
#[test]
fn test_volatility_threshold_boundary() {
    use common::{DensityInfo, TimeSeriesCharacteristics};

    let selector = AdaptiveModelSelector::default();

    // volatility = 0.14 → condition NOT met
    let chars = TimeSeriesCharacteristics {
        volatility: 0.14,
        density: DensityInfo {
            regular: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        !strategy.strategy_name.contains("volatility"),
        "vol=0.14 should NOT trigger volatility: got {}",
        strategy.strategy_name
    );

    // volatility = 0.16 → condition met
    let chars = TimeSeriesCharacteristics {
        volatility: 0.16,
        density: DensityInfo {
            regular: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("volatility"),
        "vol=0.16 should trigger volatility: got {}",
        strategy.strategy_name
    );
}

/// Test data size threshold boundary conditions.
/// Boundaries: < 100 → small_dataset, > 1000 → large_dataset, else → balanced
#[test]
fn test_data_size_boundary() {
    use common::TimeSeriesCharacteristics;

    let selector = AdaptiveModelSelector::new(Some(100), Some(1000));
    let mut chars = TimeSeriesCharacteristics::default();
    chars.density.regular = true; // no irregular strategy

    // 99 → small_dataset
    let strategy = selector.select_strategy_from_characteristics(&chars, 99, 900);
    assert_eq!(strategy.strategy_name, "small_dataset");

    // 100 → balanced (100 is outside small_dataset range)
    let strategy = selector.select_strategy_from_characteristics(&chars, 100, 900);
    assert_eq!(strategy.strategy_name, "balanced");

    // 1000 → balanced (1000 is outside large_dataset range)
    let strategy = selector.select_strategy_from_characteristics(&chars, 1000, 900);
    assert_eq!(strategy.strategy_name, "balanced");

    // 1001 → large_dataset
    let strategy = selector.select_strategy_from_characteristics(&chars, 1001, 900);
    assert_eq!(strategy.strategy_name, "large_dataset");
}

/// Test scoring competition between trend and seasonality.
/// Seasonality: score × 2.0, Trend: r² × 1.5
#[test]
fn test_trend_vs_seasonality_scoring() {
    use common::TimeSeriesCharacteristics;

    let selector = AdaptiveModelSelector::default();

    // Seasonality: score=0.5 → score = 0.5 × 2.0 = 1.0
    // Trend: r²=0.8 → score = 0.8 × 1.5 = 1.2
    // → Trend wins
    let mut chars = TimeSeriesCharacteristics::default();
    chars.seasonality.strength = "strong".into();
    chars.seasonality.score = 0.5;
    chars.trend.strength = "strong".into();
    chars.trend.r_squared = 0.8;
    chars.density.regular = true;

    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("trend"),
        "trend (score=1.2) should win over seasonal (score=1.0): got {}",
        strategy.strategy_name
    );

    // Seasonality: score=0.7 → score = 0.7 × 2.0 = 1.4
    // Trend: r²=0.8 → score = 0.8 × 1.5 = 1.2
    // → Seasonality wins
    chars.seasonality.score = 0.7;
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("seasonal"),
        "seasonal (score=1.4) should win over trend (score=1.2): got {}",
        strategy.strategy_name
    );
}

/// Test irregular vs other strong characteristics.
/// Irregular has fixed score = 1.0
#[test]
fn test_irregular_vs_strong_characteristics() {
    use common::TimeSeriesCharacteristics;

    let selector = AdaptiveModelSelector::default();

    // Irregular: fixed score = 1.0
    // Trend: r²=0.8 → score = 0.8 × 1.5 = 1.2
    // → Trend wins
    let mut chars = TimeSeriesCharacteristics::default();
    chars.density.regular = false; // irregular
    chars.trend.strength = "strong".into();
    chars.trend.r_squared = 0.8;

    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("trend"),
        "trend should win over irregular: got {}",
        strategy.strategy_name
    );

    // Irregular: fixed score = 1.0
    // Trend: r²=0.75 → score = 0.75 × 1.5 = 1.125
    // → Trend still wins
    chars.trend.r_squared = 0.75;
    let strategy = selector.select_strategy_from_characteristics(&chars, 200, 900);
    assert!(
        strategy.strategy_name.contains("trend"),
        "trend (score=1.125) should still win over irregular (score=1.0): got {}",
        strategy.strategy_name
    );

    // Irregular only (no other strong characteristics)
    let mut chars_irregular = TimeSeriesCharacteristics::default();
    chars_irregular.density.regular = false;
    let strategy = selector.select_strategy_from_characteristics(&chars_irregular, 200, 900);
    assert!(
        strategy.strategy_name.contains("irregular"),
        "irregular should be selected when no other strong characteristics: got {}",
        strategy.strategy_name
    );
}

// ============================================================================
// Strategy Content Tests
// ============================================================================

/// All strategies that may be used for seasonal data should include MSTL.
/// This is a regression test: small_dataset previously lacked MSTL, causing
/// poor ensemble performance on multi_seasonal_100.
#[test]
fn test_all_strategies_include_key_fast_models() {
    let strategies = super::initialize_strategies();

    // Strategies that should include MSTL (may encounter seasonal data)
    let strategies_needing_mstl = ["strong_seasonal", "small_dataset", "balanced"];

    for name in strategies_needing_mstl {
        let strategy = strategies
            .get(name)
            .unwrap_or_else(|| panic!("Strategy {} not found", name));
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
