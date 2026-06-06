//! Diagnostic: decompose long-horizon damping on real token JSON fixtures.
//!
//! Reads price-history fixtures from a directory (`tests/real/.tmp` by
//! default, override with `DIAG_DATA_DIR=...`), splits each series so that
//! the trailing `horizon_secs` window becomes the held-out actual, then
//! runs every individual model in isolation plus the full pipeline on the
//! training subset. The per-token output records each model's final
//! predicted return, the actual return, and whether the prediction was
//! flat / right direction / wrong direction.
//!
//! Output is one line per (token, model). Aggregate counts per model
//! follow.
//!
//! Marked `#[ignore]` so it does not run in CI by default:
//!
//! ```text
//! cargo test --test real_data_over_damping -p predictor \
//!     -- --ignored --nocapture
//! ```
//!
//! To target zaciraci-supplied fixtures place them under
//! `tests/real/.tmp/` (gitignored) or point `DIAG_DATA_DIR` at any
//! directory containing `*.json` files in the `{description, data:
//! [{timestamp, price, ...}]}` schema used by `bench::sweep::loader`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use chrono::{NaiveDateTime, TimeDelta};
use common::BigDecimal;
use common::ForecastModel;
use models::{EtsModel, MstlEtsModel, NptsModel, ThetaModel};
use num_traits::ToPrimitive;
use predictor::{PredictionInput, predict};
use serde::Deserialize;

#[derive(Deserialize)]
struct SeriesFile {
    #[serde(default)]
    data: Vec<DataPoint>,
}

#[derive(Deserialize)]
struct DataPoint {
    timestamp: String,
    price: String,
}

const FLAT_RETURN_EPSILON: f64 = 1e-7;
/// Default horizon when the env var is unset. Matches zaciraci's
/// production setting (168 hours = 1 week).
const DEFAULT_HORIZON_SECS: i64 = 168 * 3600;

fn diag_dir() -> PathBuf {
    if let Ok(p) = env::var("DIAG_DATA_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/real/.tmp")
}

fn diag_horizon_secs() -> i64 {
    env::var("DIAG_HORIZON_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_HORIZON_SECS)
}

fn load_dir(path: &Path) -> Vec<(String, BTreeMap<NaiveDateTime, BigDecimal>)> {
    let mut out = Vec::new();
    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(e) => {
            println!(
                "# no data dir at {}: {e}\n# place token JSONs there or set DIAG_DATA_DIR.",
                path.display()
            );
            return out;
        }
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|s| s == "json").unwrap_or(false))
        .collect();
    paths.sort();

    for p in paths {
        let series_id = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("series")
            .to_string();
        let bytes = match fs::read(&p) {
            Ok(b) => b,
            Err(e) => {
                println!("# skip {}: read failed: {e}", p.display());
                continue;
            }
        };
        let parsed: SeriesFile = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                println!("# skip {}: parse failed: {e}", p.display());
                continue;
            }
        };
        let mut data: BTreeMap<NaiveDateTime, BigDecimal> = BTreeMap::new();
        for pt in parsed.data {
            let ts = match NaiveDateTime::parse_from_str(&pt.timestamp, "%Y-%m-%dT%H:%M:%S%.f") {
                Ok(t) => t,
                Err(_) => continue,
            };
            let price = match BigDecimal::from_str(&pt.price) {
                Ok(d) => d,
                Err(_) => continue,
            };
            data.insert(ts, price);
        }
        if data.is_empty() {
            println!("# skip {}: empty data", p.display());
            continue;
        }
        out.push((series_id, data));
    }
    out
}

fn split_series(
    full: &BTreeMap<NaiveDateTime, BigDecimal>,
    horizon: TimeDelta,
) -> Option<(
    BTreeMap<NaiveDateTime, BigDecimal>,
    BTreeMap<NaiveDateTime, BigDecimal>,
)> {
    let &last_ts = full.keys().next_back()?;
    let cutoff = last_ts - horizon;
    let train: BTreeMap<_, _> = full
        .range(..=cutoff)
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    let actual: BTreeMap<_, _> = full
        .range(cutoff..)
        .filter(|(k, _)| **k > cutoff)
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    if train.len() < 50 || actual.is_empty() {
        return None;
    }
    Some((train, actual))
}

fn decimal_to_f64(d: &BigDecimal) -> Option<f64> {
    d.to_f64().filter(|v| v.is_finite())
}

fn last_actual_return(actual: &BTreeMap<NaiveDateTime, BigDecimal>, current: f64) -> Option<f64> {
    let last_val = decimal_to_f64(actual.values().next_back()?)?;
    if current.abs() < 1e-150 {
        return None;
    }
    Some((last_val - current) / current)
}

#[derive(Debug, Clone, Copy)]
enum Verdict {
    Flat,
    Right,
    Wrong,
    Undefined,
}

impl Verdict {
    fn classify(pred_return: f64, actual_return: Option<f64>) -> Self {
        if pred_return.abs() < FLAT_RETURN_EPSILON {
            return Verdict::Flat;
        }
        let Some(a) = actual_return else {
            return Verdict::Undefined;
        };
        if a.abs() < FLAT_RETURN_EPSILON {
            return Verdict::Undefined;
        }
        if pred_return.signum() == a.signum() {
            Verdict::Right
        } else {
            Verdict::Wrong
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Flat => "flat",
            Verdict::Right => "right",
            Verdict::Wrong => "wrong",
            Verdict::Undefined => "—",
        }
    }
}

#[derive(Default)]
struct PerModelStats {
    n: usize,
    flat: usize,
    right: usize,
    wrong: usize,
    undefined: usize,
}

impl PerModelStats {
    fn record(&mut self, v: Verdict) {
        self.n += 1;
        match v {
            Verdict::Flat => self.flat += 1,
            Verdict::Right => self.right += 1,
            Verdict::Wrong => self.wrong += 1,
            Verdict::Undefined => self.undefined += 1,
        }
    }
}

fn run_model<M: ForecastModel>(
    model: &mut M,
    train_values: &[f64],
    train_ts: &[NaiveDateTime],
    horizon_steps: usize,
) -> Option<Vec<f64>> {
    model
        .fit_predict(train_values, train_ts, horizon_steps)
        .ok()
        .map(|out| out.mean)
}

#[test]
#[ignore = "diagnostic: decompose per-model damping on real token JSONs (see DIAG_DATA_DIR)"]
fn per_model_real_token_amplitude() {
    let dir = diag_dir();
    let horizon_secs = diag_horizon_secs();
    let horizon_delta = TimeDelta::seconds(horizon_secs);

    let series = load_dir(&dir);
    if series.is_empty() {
        println!(
            "# no series loaded; supply JSON fixtures under {}",
            dir.display()
        );
        return;
    }

    println!("# DIAG_DATA_DIR = {}", dir.display());
    println!("# DIAG_HORIZON_SECS = {horizon_secs}");
    println!("# loaded series = {}", series.len());
    println!();
    println!(
        "{:<28}  {:<14}  {:>14}  {:>14}  {:>8}",
        "series_id", "model", "pred_return", "actual_return", "verdict"
    );

    let mut stats: BTreeMap<&str, PerModelStats> = BTreeMap::new();
    for name in [
        "EtsModel(None)",
        "ThetaModel",
        "MstlEtsModel(None)",
        "NptsModel",
        "FullPipeline",
    ] {
        stats.insert(name, PerModelStats::default());
    }

    for (series_id, full) in &series {
        let Some((train, actual)) = split_series(full, horizon_delta) else {
            println!("# {series_id}: split failed (insufficient train/actual)");
            continue;
        };
        let train_ts: Vec<NaiveDateTime> = train.keys().copied().collect();
        let train_values: Vec<f64> = train.values().filter_map(decimal_to_f64).collect();
        if train_values.len() != train.len() {
            println!("# {series_id}: non-finite train value, skip");
            continue;
        }
        let current = *train_values.last().unwrap();
        if current.abs() < 1e-150 {
            println!("# {series_id}: degenerate baseline {current}, skip");
            continue;
        }
        let actual_return = last_actual_return(&actual, current);

        // We pick a number of horizon steps that fits the trailing window.
        // The downstream models all return `horizon_steps` mean values; we
        // use the last one as the horizon-end forecast.
        let median_interval_secs = if train_ts.len() >= 2 {
            (train_ts[train_ts.len() - 1] - train_ts[0]).num_seconds() / (train_ts.len() as i64 - 1)
        } else {
            3600
        }
        .max(1);
        let horizon_steps = ((horizon_secs / median_interval_secs).max(1)) as usize;

        let report =
            |name: &'static str, pred_return: f64, stats: &mut BTreeMap<&str, PerModelStats>| {
                let v = Verdict::classify(pred_return, actual_return);
                stats.entry(name).or_default().record(v);
                let actual_disp = actual_return
                    .map(|a| format!("{a:>+.4}"))
                    .unwrap_or_else(|| "—".to_string());
                println!(
                    "{:<28}  {:<14}  {:>+14.4}  {:>14}  {:>8}",
                    series_id,
                    name,
                    pred_return,
                    actual_disp,
                    v.as_str(),
                );
            };

        let mut ets = EtsModel::new(None);
        if let Some(mean) = run_model(&mut ets, &train_values, &train_ts, horizon_steps) {
            let pr = (mean[mean.len() - 1] - current) / current;
            report("EtsModel(None)", pr, &mut stats);
        }
        let mut theta = ThetaModel::new();
        if let Some(mean) = run_model(&mut theta, &train_values, &train_ts, horizon_steps) {
            let pr = (mean[mean.len() - 1] - current) / current;
            report("ThetaModel", pr, &mut stats);
        }
        let mut mstl = MstlEtsModel::new(None);
        if let Some(mean) = run_model(&mut mstl, &train_values, &train_ts, horizon_steps) {
            let pr = (mean[mean.len() - 1] - current) / current;
            report("MstlEtsModel(None)", pr, &mut stats);
        }
        let mut npts = NptsModel::new(None);
        if let Some(mean) = run_model(&mut npts, &train_values, &train_ts, horizon_steps) {
            let pr = (mean[mean.len() - 1] - current) / current;
            report("NptsModel", pr, &mut stats);
        }

        let input = PredictionInput {
            data: train.clone(),
            horizon: horizon_delta,
        };
        if let Ok(result) = predict(&input) {
            let pred_values: Vec<f64> = result
                .forecast_values
                .values()
                .filter_map(decimal_to_f64)
                .collect();
            if !pred_values.is_empty() {
                let last = *pred_values.last().unwrap();
                let pr = (last - current) / current;
                report("FullPipeline", pr, &mut stats);
            }
        }
    }

    println!();
    println!("# aggregate per-model verdicts");
    println!(
        "{:<22}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
        "model", "n", "flat", "right", "wrong", "undef"
    );
    for (name, s) in &stats {
        println!(
            "{:<22}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
            name, s.n, s.flat, s.right, s.wrong, s.undefined
        );
    }
}
