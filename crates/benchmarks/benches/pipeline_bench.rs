//! End-to-end pipeline benchmarks.
//!
//! Covers: full predict() pipeline with various data patterns.

use chrono::{NaiveDate, TimeDelta};
use common::BigDecimal;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use predictor::{predict, PredictionInput};
use std::collections::BTreeMap;
use std::str::FromStr;

fn make_input(values: Vec<f64>, horizon_hours: i64) -> PredictionInput {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let data: BTreeMap<_, _> = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let ts = base + chrono::Duration::hours(i as i64);
            let decimal = BigDecimal::from_str(&format!("{:.6}", v)).unwrap();
            (ts, decimal)
        })
        .collect();

    PredictionInput {
        data,
        horizon: TimeDelta::hours(horizon_hours),
    }
}

fn generate_trend_seasonal(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
        })
        .collect()
}

fn generate_pure_trend(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + 2.5 * i as f64).collect()
}

fn generate_stationary(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = ((state >> 33) as f64) / (u32::MAX as f64);
            100.0 + (frac * 2.0 - 1.0) * 10.0
        })
        .collect()
}

fn bench_pipeline_trend_seasonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_trend_seasonal");
    group.sample_size(10); // Reduce samples for expensive pipeline

    for n in [100, 200, 500] {
        let values = generate_trend_seasonal(n, 12);
        let horizon_hours = (n / 10).max(5) as i64;
        let input = make_input(values, horizon_hours);

        group.bench_with_input(BenchmarkId::from_parameter(n), &input, |b, inp| {
            b.iter(|| predict(black_box(inp)))
        });
    }

    group.finish();
}

fn bench_pipeline_pure_trend(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_pure_trend");
    group.sample_size(10);

    for n in [100, 200, 500] {
        let values = generate_pure_trend(n);
        let horizon_hours = (n / 10).max(5) as i64;
        let input = make_input(values, horizon_hours);

        group.bench_with_input(BenchmarkId::from_parameter(n), &input, |b, inp| {
            b.iter(|| predict(black_box(inp)))
        });
    }

    group.finish();
}

fn bench_pipeline_stationary(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_stationary");
    group.sample_size(10);

    for n in [100, 200, 500] {
        let values = generate_stationary(n, 42);
        let horizon_hours = (n / 10).max(5) as i64;
        let input = make_input(values, horizon_hours);

        group.bench_with_input(BenchmarkId::from_parameter(n), &input, |b, inp| {
            b.iter(|| predict(black_box(inp)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pipeline_trend_seasonal,
    bench_pipeline_pure_trend,
    bench_pipeline_stationary
);
criterion_main!(benches);
