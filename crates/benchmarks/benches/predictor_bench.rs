//! Predictor benchmarks: pool_threads comparison.
//!
//! Measures prediction speed with different dedicated ThreadPool sizes.
//! Run: `cargo bench --bench predictor_bench`

use chrono::{NaiveDate, TimeDelta};
use common::BigDecimal;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use predictor::{PredictionInput, Predictor};
use std::str::FromStr;

/// Create benchmark input with 15-minute intervals (matching production).
fn make_input(values: Vec<f64>) -> PredictionInput {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let data = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let ts = base + chrono::Duration::minutes(15 * i as i64);
            let decimal = BigDecimal::from_str(&format!("{v:.6}")).unwrap();
            (ts, decimal)
        })
        .collect();

    PredictionInput {
        data,
        horizon: TimeDelta::hours(24),
    }
}

/// Generate trend + seasonal data with period in data points (not hours).
/// With 15-min intervals, period=96 = 1 day cycle.
fn generate_trend_seasonal(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
        })
        .collect()
}

fn bench_predictor_pool_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("predictor_pool_threads");
    group.sample_size(10);

    for n in [500, 2880] {
        let values = generate_trend_seasonal(n, 96);
        let input = make_input(values);

        for threads in [1, 2, 3] {
            let predictor = Predictor::new(threads).expect("Failed to create Predictor");
            let id = BenchmarkId::new(format!("n{n}_threads{threads}"), n);
            group.bench_with_input(id, &input, |b, inp| {
                b.iter(|| predictor.predict(black_box(inp)))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_predictor_pool_threads);
criterion_main!(benches);
