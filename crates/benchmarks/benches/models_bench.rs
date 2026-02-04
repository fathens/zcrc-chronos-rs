//! Benchmarks for forecasting models.
//!
//! Covers: NptsModel, EtsModel, ThetaModel, MstlEtsModel, SeasonalNaiveModel.

use chrono::{NaiveDate, NaiveDateTime};
use common::ForecastModel;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use models::{EtsModel, MstlEtsModel, NptsModel, SeasonalNaiveModel, ThetaModel};

fn make_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::hours(i as i64))
        .collect()
}

fn generate_trend_seasonal_data(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            100.0
                + 1.5 * i as f64
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
        })
        .collect()
}

fn bench_npts(c: &mut Criterion) {
    let mut group = c.benchmark_group("npts_fit_predict");

    for n in [100, 500, 1000] {
        let values = generate_trend_seasonal_data(n, 12);
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

fn bench_ets(c: &mut Criterion) {
    let mut group = c.benchmark_group("ets_fit_predict");

    for n in [100, 500, 1000] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = EtsModel::new(Some(12));
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

fn bench_theta(c: &mut Criterion) {
    let mut group = c.benchmark_group("theta_fit_predict");

    for n in [100, 500, 1000] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = ThetaModel::new();
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

fn bench_mstl_ets(c: &mut Criterion) {
    let mut group = c.benchmark_group("mstl_ets_fit_predict");

    for n in [100, 500, 1000] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = MstlEtsModel::new(Some(vec![12]));
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

fn bench_seasonal_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("seasonal_naive_fit_predict");

    for n in [100, 500, 1000] {
        let values = generate_trend_seasonal_data(n, 12);
        let timestamps = make_timestamps(n);
        let horizon = (n / 10).max(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(values, timestamps, horizon),
            |b, (vals, ts, h)| {
                b.iter(|| {
                    let mut model = SeasonalNaiveModel::new(Some(12));
                    model.fit_predict(black_box(vals), black_box(ts), *h)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_npts,
    bench_ets,
    bench_theta,
    bench_mstl_ets,
    bench_seasonal_naive
);
criterion_main!(benches);
