use super::*;
use chrono::NaiveDate;
use chronos_core::ForecastModel;

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

#[test]
fn test_ets_linear_trend() {
    let mut model = EtsModel::new(None);
    let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 2.0).collect();
    let ts = make_timestamps(50);
    let output = model.fit_predict(&values, &ts, 5).unwrap();
    assert_eq!(output.mean.len(), 5);
    // Should predict increasing values
    assert!(output.mean[0] > values.last().copied().unwrap_or(0.0) - 10.0);
    assert!(output.lower_quantile.is_some());
    assert!(output.upper_quantile.is_some());
}

#[test]
fn test_ets_insufficient_data() {
    let mut model = EtsModel::new(None);
    let result = model.fit_predict(&[1.0, 2.0], &make_timestamps(2), 3);
    assert!(result.is_err());
}

#[test]
fn test_ets_constant_series() {
    let mut model = EtsModel::new(None);
    let values = vec![100.0; 30];
    let ts = make_timestamps(30);
    let output = model.fit_predict(&values, &ts, 5).unwrap();
    assert_eq!(output.mean.len(), 5);
    // Constant series → predictions should be near 100
    for v in &output.mean {
        assert!(
            (*v - 100.0).abs() < 10.0,
            "Expected ~100, got {}",
            v
        );
    }
}

#[test]
fn test_ets_seasonal_additive() {
    let n = 120;
    let m = 12;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            500.0 + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin()
        })
        .collect();
    let ts = make_timestamps(n);
    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(&values, &ts, 12).unwrap();
    assert_eq!(output.mean.len(), 12);
    assert!(output.lower_quantile.is_some());
    assert!(output.upper_quantile.is_some());
}

#[test]
fn test_ets_seasonal_with_trend() {
    let n = 120;
    let m = 12;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            100.0
                + 2.0 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / m as f64).sin()
        })
        .collect();
    let ts = make_timestamps(n);
    let mut model = EtsModel::new(Some(m));
    let output = model.fit_predict(&values, &ts, 12).unwrap();
    assert_eq!(output.mean.len(), 12);
    // Should forecast upward trend
    let last_train = values.last().copied().unwrap();
    let forecast_mean: f64 =
        output.mean.iter().sum::<f64>() / output.mean.len() as f64;
    assert!(
        forecast_mean > last_train - 50.0,
        "Forecast mean ({:.1}) should be near or above last value ({:.1})",
        forecast_mean,
        last_train
    );
}

#[test]
fn test_ets_seasonal_insufficient_cycles() {
    let m = 12;
    let values = vec![1.0; 20]; // Less than 2*12=24
    let ts = make_timestamps(20);
    let mut model = EtsModel::new(Some(m));
    let result = model.fit_predict(&values, &ts, 5);
    assert!(result.is_err());
}
