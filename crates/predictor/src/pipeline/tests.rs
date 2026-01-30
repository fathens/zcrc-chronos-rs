use super::*;
use chrono::NaiveDate;
use num_traits::{FromPrimitive, ToPrimitive};

fn make_data(n: usize) -> BTreeMap<NaiveDateTime, BigDecimal> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    (0..n)
        .map(|i| {
            let ts = base + TimeDelta::hours(i as i64);
            let val = BigDecimal::from_f64(100.0 + i as f64 * 2.0).unwrap();
            (ts, val)
        })
        .collect()
}

fn make_data_with_values(values: &[f64]) -> BTreeMap<NaiveDateTime, BigDecimal> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let ts = base + TimeDelta::hours(i as i64);
            let val = BigDecimal::from_f64(v).unwrap();
            (ts, val)
        })
        .collect()
}

#[test]
fn test_predict_uptrend() {
    let input = PredictionInput {
        data: make_data(100),
        horizon: TimeDelta::hours(10),
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
    let input = PredictionInput {
        data: make_data_with_values(&vec![42.0; 50]),
        horizon: TimeDelta::hours(5),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 5);
    // Flat data → predictions near 42
    for v in &result.forecast_values {
        let f = v.to_f64().unwrap();
        assert!((f - 42.0).abs() < 20.0, "Expected ~42, got {}", f);
    }
}

#[test]
fn test_predict_seasonal() {
    let n = 120;
    let values: Vec<f64> = (0..n)
        .map(|i| 500.0 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() * 50.0)
        .collect();
    let input = PredictionInput {
        data: make_data_with_values(&values),
        horizon: TimeDelta::hours(12),
    };

    let result = predict(&input).unwrap();
    assert_eq!(result.forecast_values.len(), 12);
}

#[test]
fn test_predict_validation_errors() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Too few points
    let result = predict(&PredictionInput {
        data: [(base, BigDecimal::from(1))].into_iter().collect(),
        horizon: TimeDelta::hours(5),
    });
    assert!(result.is_err());

    // Zero horizon
    let result = predict(&PredictionInput {
        data: make_data_with_values(&[1.0; 10]),
        horizon: TimeDelta::zero(),
    });
    assert!(result.is_err());

    // Negative horizon
    let result = predict(&PredictionInput {
        data: make_data_with_values(&[1.0; 10]),
        horizon: TimeDelta::hours(-1),
    });
    assert!(result.is_err());
}

#[test]
fn test_horizon_to_steps() {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Hourly data: 24 hours → 24 steps
    let hourly_ts: Vec<_> = (0..100).map(|i| base + TimeDelta::hours(i)).collect();
    assert_eq!(horizon_to_steps(&TimeDelta::hours(24), &hourly_ts), 24);

    // Daily data: 7 days → 7 steps
    let daily_ts: Vec<_> = (0..30).map(|i| base + TimeDelta::days(i)).collect();
    assert_eq!(horizon_to_steps(&TimeDelta::days(7), &daily_ts), 7);

    // Edge case: fewer than 2 timestamps → return 1
    let single_ts = vec![base];
    assert_eq!(horizon_to_steps(&TimeDelta::hours(24), &single_ts), 1);
}

#[test]
fn test_calculate_time_budget() {
    // 100 points → 60 + 10 = 70s
    assert!((calculate_time_budget(100) - 70.0).abs() < 0.1);

    // 500 points → 60 + 50 = 110s
    assert!((calculate_time_budget(500) - 110.0).abs() < 0.1);

    // 10000 points → capped at 900s
    assert!((calculate_time_budget(10000) - 900.0).abs() < 0.1);
}
