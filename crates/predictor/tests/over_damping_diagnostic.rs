//! Diagnostic: measure long-horizon amplitude on each individual model and
//! through the full pipeline, on a "weak trend + small noise" synthetic
//! series that mimics the production crypto-token data.
//!
//! Marked `#[ignore]` so it does not run in CI by default; invoke with
//!
//! ```text
//! cargo test --test over_damping_diagnostic -- --ignored --nocapture
//! ```
//!
//! Hypothesis being checked (improve.md, 2026-06-04 verification): the
//! pipeline over-damps multi-step extrapolation at the production horizon
//! (168h), driving 79% of predictions flat. The investigator
//! report flagged three candidate sources:
//!   1. augurs `AutoETS("ZZN")` selecting ETS(A,N,N) on noisy data
//!      (shared by EtsModel, MstlEtsModel trend component, ThetaModel
//!      theta-2 line).
//!   2. Theta's `(linear + theta2) / 2` blending halves amplitude.
//!   3. softmax_ensemble temperature=0.5 + MASE floor=2.0 prevent the
//!      ensemble from differentiating flat from trending forecasts.
//!
//! This file confirms which models exhibit damping in isolation and how
//! the full pipeline composes them.

use std::collections::BTreeMap;

use chrono::{NaiveDate, NaiveDateTime, TimeDelta};
use common::{BigDecimal, ForecastModel};
use models::{EtsModel, MstlEtsModel, NptsModel, ThetaModel};
use num_traits::FromPrimitive;
use predictor::{PredictionInput, predict};

/// Hourly samples, 30 days of weak upward drift overlaid with small
/// uniform noise. Matches "condition A" from the investigation report.
///
/// - `n` samples
/// - linear drift `slope_per_hour`
/// - relative noise band `noise_amp` (multiplicative on a 1.0 baseline)
fn synth_weak_trend(n: usize, slope_per_hour: f64, noise_amp: f64) -> Vec<f64> {
    let mut state: u64 = 42;
    (0..n)
        .map(|i| {
            // tiny LCG for repeatable noise without an extra dep
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            100.0 * (1.0 + slope_per_hour * i as f64 + (u - 0.5) * 2.0 * noise_amp)
        })
        .collect()
}

fn synth_timestamps(n: usize) -> Vec<NaiveDateTime> {
    let base = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    (0..n).map(|i| base + TimeDelta::hours(i as i64)).collect()
}

fn synth_btreemap(
    values: &[f64],
    timestamps: &[NaiveDateTime],
) -> BTreeMap<NaiveDateTime, BigDecimal> {
    timestamps
        .iter()
        .zip(values)
        .map(|(ts, &v)| (*ts, BigDecimal::from_f64(v).unwrap()))
        .collect()
}

/// Linear extrapolation amplitude that *would* be observed at the given
/// horizon if the model preserved the training trend perfectly.
fn expected_undamped_amplitude(values: &[f64], horizon: usize) -> f64 {
    let n = values.len() as f64;
    let last = values[values.len() - 1];
    let first = values[0];
    let slope = (last - first) / (n - 1.0);
    slope * horizon as f64
}

#[derive(Debug)]
struct AmplitudeRow {
    model: &'static str,
    first: f64,
    last: f64,
    amplitude: f64,
    /// `amplitude / expected_undamped` — 1.0 means the model carries the
    /// full training trend; values near 0 indicate "flat" predictions.
    damping_ratio: f64,
}

fn measure(model_name: &'static str, forecast: &[f64], expected_undamped: f64) -> AmplitudeRow {
    let first = forecast[0];
    let last = *forecast.last().unwrap();
    let amplitude = last - first;
    let damping_ratio = if expected_undamped.abs() > f64::EPSILON {
        amplitude / expected_undamped
    } else {
        f64::NAN
    };
    AmplitudeRow {
        model: model_name,
        first,
        last,
        amplitude,
        damping_ratio,
    }
}

#[test]
#[ignore = "diagnostic: print per-model long-horizon amplitude"]
fn per_model_long_horizon_amplitude() {
    // 30 day hourly history (n=720), forecast h=168 (=1 week).
    // slope_per_hour = 0.0005 / 24 ≈ 2.08e-5 → 30-day total drift +1.5%.
    let n = 720usize;
    let horizon = 168usize;
    let slope_per_hour = 0.0005 / 24.0;
    let noise_amp = 0.001;

    let values = synth_weak_trend(n, slope_per_hour, noise_amp);
    let timestamps = synth_timestamps(n);
    let expected = expected_undamped_amplitude(&values, horizon);

    println!("# synthetic series (n={n}, h={horizon})");
    println!(
        "  train_first = {:.6}, train_last = {:.6}, observed_slope = {:.6e}",
        values[0],
        values[n - 1],
        (values[n - 1] - values[0]) / (n as f64 - 1.0),
    );
    println!("  expected undamped amplitude over h={horizon}: {expected:.6}");
    println!();

    let mut rows: Vec<AmplitudeRow> = Vec::new();

    // EtsModel (non-seasonal) — directly tests hypothesis #1.
    let mut ets = EtsModel::new(None);
    let out = ets
        .fit_predict(&values, &timestamps, horizon)
        .expect("ETS fit_predict");
    rows.push(measure("EtsModel", &out.mean, expected));

    // ThetaModel — directly tests hypothesis #2.
    let mut theta = ThetaModel::new();
    let out = theta
        .fit_predict(&values, &timestamps, horizon)
        .expect("Theta fit_predict");
    rows.push(measure("ThetaModel", &out.mean, expected));

    // MstlEtsModel — tests whether MSTL's trend component shares the
    // same damping (its trend line is also AutoETS("ZZN")).
    let mut mstl = MstlEtsModel::new(None);
    let out = mstl
        .fit_predict(&values, &timestamps, horizon)
        .expect("MSTL fit_predict");
    rows.push(measure("MstlEtsModel", &out.mean, expected));

    // NptsModel — k-NN baseline; expected to be flat regardless.
    let mut npts = NptsModel::new(None);
    let out = npts
        .fit_predict(&values, &timestamps, horizon)
        .expect("NPTS fit_predict");
    rows.push(measure("NptsModel", &out.mean, expected));

    // Full pipeline (Predictor::predict) — combines the above via
    // softmax_ensemble. Tests hypothesis #3 by comparing the ensemble
    // amplitude to each individual model.
    let input = PredictionInput {
        data: synth_btreemap(&values, &timestamps),
        horizon: TimeDelta::hours(horizon as i64),
    };
    let result = predict(&input).expect("Predictor::predict");
    let ensemble: Vec<f64> = result
        .forecast_values
        .values()
        .map(|d| {
            use num_traits::ToPrimitive;
            d.to_f64().unwrap()
        })
        .collect();
    rows.push(measure("FullPipeline", &ensemble, expected));

    println!(
        "{:<14}  {:>12}  {:>12}  {:>12}  {:>14}",
        "model", "first", "last", "amplitude", "damping_ratio"
    );
    for r in &rows {
        println!(
            "{:<14}  {:>12.6}  {:>12.6}  {:>12.6}  {:>14.6}",
            r.model, r.first, r.last, r.amplitude, r.damping_ratio,
        );
    }
}

#[test]
#[ignore = "diagnostic: sweep SNR — does damping appear when noise overwhelms the trend?"]
fn full_pipeline_amplitude_per_snr() {
    // Hold the trend constant, sweep the noise band. At low SNR
    // (noise_amp ≫ slope × n) AICc should start preferring ETS(A,N,N) =
    // flat. If that is the source of the production "flat 79 %" pattern,
    // we expect `damping_ratio` to collapse toward 0 as noise rises.
    let n = 720usize;
    let horizon = 168usize;
    let slope_per_hour = 0.0005 / 24.0; // identical across regimes
    let timestamps = synth_timestamps(n);

    println!(
        "# slope/hr = {slope_per_hour:.3e}, n = {n}, h = {horizon}, observed amplitude ÷ undamped expectation"
    );
    println!(
        "{:>12}  {:>12}  {:>12}  {:>14}",
        "noise_amp", "expected", "amplitude", "damping_ratio"
    );
    for &noise_amp in &[0.0001_f64, 0.001, 0.005, 0.01, 0.02, 0.05] {
        let values = synth_weak_trend(n, slope_per_hour, noise_amp);
        let expected = expected_undamped_amplitude(&values, horizon);
        let input = PredictionInput {
            data: synth_btreemap(&values, &timestamps),
            horizon: TimeDelta::hours(horizon as i64),
        };
        let result = predict(&input).expect("Predictor::predict");
        let ensemble: Vec<f64> = result
            .forecast_values
            .values()
            .map(|d| {
                use num_traits::ToPrimitive;
                d.to_f64().unwrap()
            })
            .collect();
        let amplitude = ensemble.last().unwrap() - ensemble[0];
        let damping_ratio = if expected.abs() > f64::EPSILON {
            amplitude / expected
        } else {
            f64::NAN
        };
        println!("{noise_amp:>12.5}  {expected:>12.6}  {amplitude:>12.6}  {damping_ratio:>14.6}");
    }
}

#[test]
#[ignore = "diagnostic: localise the flat-collapse cliff and the per-model contribution at the cliff"]
fn flat_collapse_breakdown() {
    let n = 720usize;
    let horizon = 168usize;
    let slope_per_hour = 0.0005 / 24.0;
    let timestamps = synth_timestamps(n);

    println!("# pipeline amplitude near the flat-collapse cliff");
    println!(
        "{:>12}  {:>12}  {:>14}",
        "noise_amp", "amplitude", "damping_ratio"
    );
    for &noise_amp in &[
        0.003_f64, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007,
    ] {
        let values = synth_weak_trend(n, slope_per_hour, noise_amp);
        let expected = expected_undamped_amplitude(&values, horizon);
        let input = PredictionInput {
            data: synth_btreemap(&values, &timestamps),
            horizon: TimeDelta::hours(horizon as i64),
        };
        let result = predict(&input).expect("Predictor::predict");
        let ensemble: Vec<f64> = result
            .forecast_values
            .values()
            .map(|d| {
                use num_traits::ToPrimitive;
                d.to_f64().unwrap()
            })
            .collect();
        let amplitude = ensemble.last().unwrap() - ensemble[0];
        let damping_ratio = if expected.abs() > f64::EPSILON {
            amplitude / expected
        } else {
            f64::NAN
        };
        println!("{noise_amp:>12.5}  {amplitude:>12.6}  {damping_ratio:>14.6}");
    }

    println!();
    println!("# per-model amplitude at the cliff (noise_amp = 0.005)");
    let values = synth_weak_trend(n, slope_per_hour, 0.005);
    let expected = expected_undamped_amplitude(&values, horizon);
    let analyzer = analyzer::TimeSeriesAnalyzer::new();
    let chars = analyzer.analyze(&values, &timestamps);
    let detected_season = chars.seasonality.period;
    println!("  expected undamped amplitude: {expected:.6}");
    println!(
        "  detected seasonality: period={:?}, regime={:?}, trend.slope={:.3e}",
        detected_season, chars.regime.regime, chars.trend.slope
    );
    println!(
        "{:<22}  {:>12}  {:>12}  {:>14}",
        "model", "first", "last", "damping_ratio"
    );

    // EtsModel with the analyzer-detected season_period — this is the
    // same model the trainer instantiates internally.
    let mut ets_pipeline = EtsModel::new(detected_season);
    let out = ets_pipeline
        .fit_predict(&values, &timestamps, horizon)
        .unwrap();
    let r = measure("EtsModel(season=auto)", &out.mean, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );

    let mut ets_none = EtsModel::new(None);
    let out = ets_none.fit_predict(&values, &timestamps, horizon).unwrap();
    let r = measure("EtsModel(season=None)", &out.mean, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );

    let mut theta = ThetaModel::new();
    let out = theta.fit_predict(&values, &timestamps, horizon).unwrap();
    let r = measure("ThetaModel", &out.mean, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );

    let periods_for_mstl = detected_season.map(|p| vec![p]);
    let mut mstl_pipeline = MstlEtsModel::new(periods_for_mstl);
    let out = mstl_pipeline
        .fit_predict(&values, &timestamps, horizon)
        .unwrap();
    let r = measure("MstlEtsModel(seas=auto)", &out.mean, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );

    let mut mstl_none = MstlEtsModel::new(None);
    let out = mstl_none
        .fit_predict(&values, &timestamps, horizon)
        .unwrap();
    let r = measure("MstlEtsModel(seas=None)", &out.mean, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );

    let mut npts = NptsModel::new(None);
    let out = npts.fit_predict(&values, &timestamps, horizon).unwrap();
    let r = measure("NptsModel", &out.mean, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );

    let input = PredictionInput {
        data: synth_btreemap(&values, &timestamps),
        horizon: TimeDelta::hours(horizon as i64),
    };
    let result = predict(&input).unwrap();
    let ensemble: Vec<f64> = result
        .forecast_values
        .values()
        .map(|d| {
            use num_traits::ToPrimitive;
            d.to_f64().unwrap()
        })
        .collect();
    let r = measure("FullPipeline", &ensemble, expected);
    println!(
        "{:<22}  {:>12.6}  {:>12.6}  {:>14.6}",
        r.model, r.first, r.last, r.damping_ratio
    );
    println!("  strategy_name = {}", result.strategy_name);
    println!("  model_name = {}", result.model_name);
    println!("  model_count = {}", result.model_count);
}

#[test]
#[ignore = "diagnostic: print per-horizon ensemble amplitude across {24,72,168,360} h"]
fn full_pipeline_amplitude_per_horizon() {
    // Same synthetic regime as the per-model test but sweep the
    // production-relevant horizons. A monotonically shrinking
    // damping_ratio with horizon is the smoking gun that the
    // multi-step extrapolation collapses toward flat.
    let n = 720usize;
    let slope_per_hour = 0.0005 / 24.0;
    let noise_amp = 0.001;
    let values = synth_weak_trend(n, slope_per_hour, noise_amp);
    let timestamps = synth_timestamps(n);

    println!(
        "# synthetic series (n={n}), slope/hr = {slope_per_hour:.3e}, noise_amp = {noise_amp}"
    );
    println!(
        "{:>8}  {:>12}  {:>14}",
        "horizon", "amplitude", "damping_ratio"
    );
    for &h in &[24usize, 72, 168, 360] {
        let input = PredictionInput {
            data: synth_btreemap(&values, &timestamps),
            horizon: TimeDelta::hours(h as i64),
        };
        let result = predict(&input).expect("Predictor::predict");
        let ensemble: Vec<f64> = result
            .forecast_values
            .values()
            .map(|d| {
                use num_traits::ToPrimitive;
                d.to_f64().unwrap()
            })
            .collect();
        let amplitude = ensemble.last().unwrap() - ensemble[0];
        let expected = expected_undamped_amplitude(&values, h);
        let damping_ratio = if expected.abs() > f64::EPSILON {
            amplitude / expected
        } else {
            f64::NAN
        };
        println!("{h:>8}  {amplitude:>12.6}  {damping_ratio:>14.6}");
    }
}
