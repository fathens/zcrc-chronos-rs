use super::*;
use chrono::NaiveDate;
use num_traits::{FromPrimitive, ToPrimitive};

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn to_decimals(vals: &[f64]) -> Vec<BigDecimal> {
    vals.iter()
        .map(|&v| BigDecimal::from_f64(v).unwrap())
        .collect()
}

#[test]
fn test_predict_uptrend() {
    let n = 100;
    let values: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 2.0).collect();
    let input = PredictionInput {
        timestamps: make_timestamps(n),
        values: to_decimals(&values),
        horizon: 10,
        time_budget_secs: Some(60.0),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 10);
    assert!(!result.model_name.is_empty());
    assert!(!result.strategy_name.is_empty());
    assert!(result.processing_time_secs > 0.0);
    assert!(result.model_count > 0);
}

#[test]
fn test_predict_flat() {
    let n = 50;
    let input = PredictionInput {
        timestamps: make_timestamps(n),
        values: to_decimals(&vec![42.0; n]),
        horizon: 5,
        time_budget_secs: Some(30.0),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 5);
    // Flat data → predictions near 42
    for v in &result.forecast_values {
        let f = v.to_f64().unwrap();
        assert!(
            (f - 42.0).abs() < 20.0,
            "Expected ~42, got {}",
            f
        );
    }
}

#[test]
fn test_predict_seasonal() {
    let n = 120;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            500.0 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() * 50.0
        })
        .collect();
    let input = PredictionInput {
        timestamps: make_timestamps(n),
        values: to_decimals(&values),
        horizon: 12,
        time_budget_secs: Some(60.0),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 12);
}

#[test]
fn test_predict_validation_errors() {
    // Mismatched lengths
    let result = predict(&PredictionInput {
        timestamps: make_timestamps(3),
        values: to_decimals(&[1.0, 2.0]),
        horizon: 5,
        time_budget_secs: None,
    });
    assert!(result.is_err());

    // Too few points
    let result = predict(&PredictionInput {
        timestamps: make_timestamps(1),
        values: to_decimals(&[1.0]),
        horizon: 5,
        time_budget_secs: None,
    });
    assert!(result.is_err());

    // Zero horizon
    let result = predict(&PredictionInput {
        timestamps: make_timestamps(10),
        values: to_decimals(&vec![1.0; 10]),
        horizon: 0,
        time_budget_secs: None,
    });
    assert!(result.is_err());
}
