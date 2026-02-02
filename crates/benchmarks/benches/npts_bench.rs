//! Detailed benchmarks for NPTS (Non-Parametric Time Series) model.
//!
//! Covers: K-nearest neighbor distance calculations, various K values, data sizes.

use chrono::{NaiveDate, NaiveDateTime};
use common::ForecastModel;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use models::NptsModel;

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn generate_repeating_pattern(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 10) as f64 * 5.0 + 100.0).collect()
}

fn generate_sinusoidal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 100.0 + 30.0 * (i as f64 * 0.3).sin())
        .collect()
}

fn bench_npts_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("npts_varying_k");

    let n = 500;
    let values = generate_repeating_pattern(n);
    let timestamps = make_timestamps(n);
    let horizon = 50;

    for k in [1, 3, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(k),
            &(values.clone(), timestamps.clone(), horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = NptsModel::new(Some(k));
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

fn bench_npts_varying_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("npts_varying_size");

    for n in [100, 500, 1000, 2000, 5000] {
        let values = generate_sinusoidal(n);
        let timestamps = make_timestamps(n);
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = NptsModel::new(Some(5));
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

fn bench_npts_varying_horizon(c: &mut Criterion) {
    let mut group = c.benchmark_group("npts_varying_horizon");

    let n = 500;
    let values = generate_sinusoidal(n);
    let timestamps = make_timestamps(n);

    for horizon in [10, 25, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(horizon),
            &(values.clone(), timestamps.clone(), horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = NptsModel::new(Some(5));
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_npts_varying_k,
    bench_npts_varying_size,
    bench_npts_varying_horizon
);
criterion_main!(benches);
