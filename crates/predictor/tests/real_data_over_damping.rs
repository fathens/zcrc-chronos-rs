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

#[derive(Debug, Clone)]
struct TokenRow {
    series_id: String,
    regime: &'static str,
    actual_return: Option<f64>,
    ets: Option<f64>,
    theta: Option<f64>,
    mstl: Option<f64>,
    npts: Option<f64>,
    pipeline: Option<f64>,
    // Last-step predicted_std divided by `current` (price-relative scale).
    // Used by the flat-vs-non-flat diagnostic to validate the Markowitz
    // assumption "flat prediction ⇒ wide band ⇒ large variance ⇒ down-weight".
    predicted_std_ratio: Option<f64>,
}

const MODEL_LABELS: [&str; 4] = ["EtsModel", "ThetaModel", "MstlEtsModel", "NptsModel"];

fn regime_label(values: &[f64]) -> &'static str {
    use common::TimeSeriesRegime;
    let analyzer = analyzer::TimeSeriesAnalyzer::new();
    match analyzer.detect_regime(values).regime {
        TimeSeriesRegime::Trending => "Trending",
        TimeSeriesRegime::RandomWalk => "RandomWalk",
        TimeSeriesRegime::MeanReverting => "MeanReverting",
    }
}

fn collect_token_rows(
    series: &[(String, BTreeMap<NaiveDateTime, BigDecimal>)],
    horizon_secs: i64,
) -> Vec<TokenRow> {
    let horizon_delta = TimeDelta::seconds(horizon_secs);
    let mut rows = Vec::new();

    for (series_id, full) in series {
        let Some((train, actual)) = split_series(full, horizon_delta) else {
            continue;
        };
        let train_ts: Vec<NaiveDateTime> = train.keys().copied().collect();
        let train_values: Vec<f64> = train.values().filter_map(decimal_to_f64).collect();
        if train_values.len() != train.len() {
            continue;
        }
        let current = *train_values.last().unwrap();
        if current.abs() < 1e-150 {
            continue;
        }
        let actual_return = last_actual_return(&actual, current);
        let regime = regime_label(&train_values);

        let median_interval_secs = if train_ts.len() >= 2 {
            (train_ts[train_ts.len() - 1] - train_ts[0]).num_seconds() / (train_ts.len() as i64 - 1)
        } else {
            3600
        }
        .max(1);
        let horizon_steps = ((horizon_secs / median_interval_secs).max(1)) as usize;

        let make_return = |mean: Vec<f64>| -> Option<f64> {
            let last = *mean.last()?;
            Some((last - current) / current)
        };

        let mut ets_model = EtsModel::new(None);
        let ets = run_model(&mut ets_model, &train_values, &train_ts, horizon_steps)
            .and_then(make_return);

        let mut theta_model = ThetaModel::new();
        let theta = run_model(&mut theta_model, &train_values, &train_ts, horizon_steps)
            .and_then(make_return);

        let mut mstl_model = MstlEtsModel::new(None);
        let mstl = run_model(&mut mstl_model, &train_values, &train_ts, horizon_steps)
            .and_then(make_return);

        let mut npts_model = NptsModel::new(None);
        let npts = run_model(&mut npts_model, &train_values, &train_ts, horizon_steps)
            .and_then(make_return);

        let input = PredictionInput {
            data: train.clone(),
            horizon: horizon_delta,
        };
        let pipeline_result = predict(&input).ok();
        let pipeline = pipeline_result.as_ref().and_then(|result| {
            let pred_values: Vec<f64> = result
                .forecast_values
                .values()
                .filter_map(decimal_to_f64)
                .collect();
            let last = *pred_values.last()?;
            Some((last - current) / current)
        });
        let predicted_std_ratio = pipeline_result.as_ref().and_then(|result| {
            let std_map = result.predicted_std.as_ref()?;
            let std_values: Vec<f64> = std_map.values().filter_map(decimal_to_f64).collect();
            let last = *std_values.last()?;
            let denom = current.abs();
            if denom < 1e-150 {
                return None;
            }
            Some(last / denom)
        });

        rows.push(TokenRow {
            series_id: series_id.clone(),
            regime,
            actual_return,
            ets,
            theta,
            mstl,
            npts,
            pipeline,
            predicted_std_ratio,
        });
    }
    rows
}

/// Identify the individual model whose `pred_return` is largest for the
/// given row — the model "pulling" the ensemble up. `None` when no
/// individual model produced a usable prediction.
fn top_puller(row: &TokenRow) -> Option<&'static str> {
    let candidates: [(&'static str, Option<f64>); 4] = [
        ("EtsModel", row.ets),
        ("ThetaModel", row.theta),
        ("MstlEtsModel", row.mstl),
        ("NptsModel", row.npts),
    ];
    candidates
        .iter()
        .filter_map(|(name, v)| v.map(|x| (*name, x)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(name, _)| name)
}

fn decile_dir_acc(rows: &[&TokenRow]) -> (usize, usize) {
    let mut right = 0usize;
    let mut total = 0usize;
    for r in rows {
        let (Some(p), Some(a)) = (r.pipeline, r.actual_return) else {
            continue;
        };
        if p.abs() < FLAT_RETURN_EPSILON || a.abs() < FLAT_RETURN_EPSILON {
            continue;
        }
        total += 1;
        if p.signum() == a.signum() {
            right += 1;
        }
    }
    (right, total)
}

fn decile_mean_actual(rows: &[&TokenRow]) -> Option<f64> {
    let xs: Vec<f64> = rows.iter().filter_map(|r| r.actual_return).collect();
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

#[test]
#[ignore = "diagnostic: decompose the top-decile predictions that drive the production ER-selection loss"]
fn top_decile_decomposition() {
    let dir = diag_dir();
    let horizon_secs = diag_horizon_secs();
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
    println!();

    let rows = collect_token_rows(&series, horizon_secs);
    println!("# tokens collected = {}", rows.len());

    // Sort by pipeline pred_return descending; rows without a pipeline
    // prediction sink to the bottom.
    let mut ranked: Vec<&TokenRow> = rows.iter().collect();
    ranked.sort_by(|a, b| {
        let av = a.pipeline.unwrap_or(f64::NEG_INFINITY);
        let bv = b.pipeline.unwrap_or(f64::NEG_INFINITY);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = ranked.iter().filter(|r| r.pipeline.is_some()).count();
    if n < 10 {
        println!("# only {n} ranked predictions; need ≥10 for decile analysis");
        return;
    }
    let decile = (n / 10).max(1);
    let top: Vec<&TokenRow> = ranked.iter().take(decile).copied().collect();
    let bottom: Vec<&TokenRow> = ranked
        .iter()
        .rev()
        .filter(|r| r.pipeline.is_some())
        .take(decile)
        .copied()
        .collect();

    let print_decile = |label: &str, rows: &[&TokenRow]| {
        println!("\n# {label} decile (n={})", rows.len());
        println!(
            "{:<28}  {:<14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:<12}",
            "series_id", "regime", "ets", "theta", "mstl", "npts", "pipeline", "actual", "puller",
        );
        for r in rows {
            let f = |o: Option<f64>| {
                o.map(|v| format!("{v:>+10.4}"))
                    .unwrap_or_else(|| "       —".to_string())
            };
            let puller = top_puller(r).unwrap_or("—");
            println!(
                "{:<28}  {:<14}  {}  {}  {}  {}  {}  {}  {:<12}",
                r.series_id,
                r.regime,
                f(r.ets),
                f(r.theta),
                f(r.mstl),
                f(r.npts),
                f(r.pipeline),
                f(r.actual_return),
                puller,
            );
        }
        let (right, total) = decile_dir_acc(rows);
        let mean_actual = decile_mean_actual(rows);
        println!(
            "# {label} decile DirAcc = {} / {}{}",
            right,
            total,
            total
                .checked_div(1)
                .filter(|_| total > 0)
                .map(|_| format!(" ({:.1}%)", 100.0 * right as f64 / total as f64))
                .unwrap_or_default(),
        );
        if let Some(m) = mean_actual {
            println!("# {label} decile mean actual_return = {m:>+.4}");
        }

        let mut regime_counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut puller_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for r in rows {
            *regime_counts.entry(r.regime).or_default() += 1;
            if let Some(p) = top_puller(r) {
                *puller_counts.entry(p).or_default() += 1;
            }
        }
        println!("# {label} decile regime composition:");
        for (k, v) in &regime_counts {
            println!("    {k:<14} {v}");
        }
        println!("# {label} decile top puller (individual model with highest pred_return):");
        for label in MODEL_LABELS {
            let count = puller_counts.get(label).copied().unwrap_or(0);
            println!("    {label:<14} {count}");
        }
    };

    print_decile("TOP", &top);
    print_decile("BOTTOM", &bottom);
}

/// Bucket the linear-trend "span ratio" that the production detrend gate
/// uses (`|slope| × (n - 1) / |current_price|`). Boundaries are aligned
/// with the production gate (`pipeline::DETREND_SPAN_RATIO_THRESHOLD =
/// 0.15`) so that "narrow band" callers can quantify the share of flat
/// predictions whose underlying series carries a *genuine* linear trend
/// but is rejected by the gate just below the cutoff.
fn span_ratio_bucket(span_ratio: f64) -> &'static str {
    if span_ratio < 0.01 {
        // No measurable linear trend at all: detrend has nothing to
        // inject, so the resulting flat prediction is structurally
        // correct (memecoin-style "no signal" case).
        "span≈0 (no-trend)"
    } else if span_ratio < 0.10 {
        "span∈[0.01,0.10) (low-trend)"
    } else if span_ratio <= 0.15 {
        // The genuine rescue candidates: a measurable trend exists but
        // the production gate (0.15) rejects it just below the cutoff.
        "span∈[0.10,0.15] (narrow-band, rescue candidate)"
    } else {
        "span>0.15 (above gate)"
    }
}

/// Categorical outcome of the adaptive-detrend gate, recomputed locally
/// from the analyzer's public characteristics so the diagnostic does not
/// need to reach into `pipeline::compute_detrend_state` (which is
/// `pub(crate)`). The conditions are kept aligned with
/// `pipeline::compute_detrend_state`; if that gate ever changes, this
/// helper must be updated as well.
fn detrend_outcome(train_values: &[f64]) -> (&'static str, Option<f64>) {
    use common::TimeSeriesRegime;

    // Mirror `pipeline::adaptive_detrend_enabled`. The diagnostic respects
    // the same env-var override the production pipeline reads so that
    // running with `CHRONOS_ADAPTIVE_DETREND=0` reports every row as
    // env-disabled, matching real behaviour.
    let env_enabled = match std::env::var("CHRONOS_ADAPTIVE_DETREND").ok().as_deref() {
        Some(v) => {
            !(v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no"))
        }
        None => true,
    };
    if !env_enabled {
        return ("env disabled", None);
    }

    if train_values.is_empty() {
        return ("empty input", None);
    }

    let analyzer = analyzer::TimeSeriesAnalyzer::new();
    // The diagnostic reuses the raw (non-normalised) training values;
    // production runs `normalize_time_series_data` first but the fixtures
    // used here are already evenly-spaced hourly series, so the
    // normalisation is a no-op and the analysis would match.
    let timestamps: Vec<NaiveDateTime> = (0..train_values.len())
        .map(|i| {
            chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                + TimeDelta::hours(i as i64)
        })
        .collect();
    let characteristics = analyzer.analyze(train_values, &timestamps);

    // The pipeline checks `is_exponential && all values > 0` before
    // log-transforming. When the log transform is applied the detrend
    // gate short-circuits to `None`, so we report that case explicitly.
    let is_exponential = characteristics.trend.is_exponential;
    if is_exponential && train_values.iter().all(|&v| v > 0.0) {
        return ("log-transformed (gate skipped)", None);
    }

    if characteristics.regime.regime == TimeSeriesRegime::MeanReverting {
        return ("regime=MeanReverting", None);
    }
    let slope = characteristics.trend.slope;
    let intercept = characteristics.trend.intercept;
    if !slope.is_finite() || !intercept.is_finite() {
        return ("slope/intercept non-finite", None);
    }
    let current_price = *train_values.last().unwrap();
    if !current_price.is_finite() || current_price.abs() < 1e-150 {
        return ("current_price below floor", Some(0.0));
    }
    let span = slope.abs() * (train_values.len() as f64 - 1.0);
    let span_ratio = span / current_price.abs();
    if span_ratio <= 0.15 {
        return (span_ratio_bucket(span_ratio), Some(span_ratio));
    }
    ("applied (span>0.15)", Some(span_ratio))
}

#[test]
#[ignore = "diagnostic: decompose flat pipeline predictions by detrend-gate outcome"]
fn flat_prediction_decomposition() {
    let dir = diag_dir();
    let horizon_secs = diag_horizon_secs();
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
    println!();

    let rows = collect_token_rows(&series, horizon_secs);
    let total = rows.iter().filter(|r| r.pipeline.is_some()).count();
    if total == 0 {
        println!("# no rows with pipeline predictions");
        return;
    }
    println!("# tokens with pipeline prediction = {total}");

    // The 1e-7 threshold matches `pipeline::FLAT_RETURN_EPSILON`, the
    // production-side gate that classifies a pred_return as "flat".
    let flat_rows: Vec<&TokenRow> = rows
        .iter()
        .filter(|r| {
            r.pipeline
                .map(|p| p.abs() < FLAT_RETURN_EPSILON)
                .unwrap_or(false)
        })
        .collect();

    let flat_share = 100.0 * flat_rows.len() as f64 / total as f64;
    println!(
        "# flat pipeline predictions = {} / {} ({:.1}%)",
        flat_rows.len(),
        total,
        flat_share,
    );

    // Per-base-model flat counts on the same fixture universe.
    let mut per_model_flat: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for r in &rows {
        if r.pipeline.is_none() {
            continue;
        }
        for (label, opt) in [
            ("EtsModel", r.ets),
            ("ThetaModel", r.theta),
            ("MstlEtsModel", r.mstl),
            ("NptsModel", r.npts),
            ("FullPipeline", r.pipeline),
        ] {
            let entry = per_model_flat.entry(label).or_default();
            if let Some(v) = opt {
                entry.1 += 1;
                if v.abs() < FLAT_RETURN_EPSILON {
                    entry.0 += 1;
                }
            }
        }
    }
    println!("\n# per-model flat rate (over rows where the model returned a prediction)");
    println!(
        "{:<14}  {:>8}  {:>8}  {:>8}",
        "model", "flat", "total", "rate"
    );
    for (label, (flat, n)) in &per_model_flat {
        let rate = if *n > 0 {
            format!("{:>7.1}%", 100.0 * (*flat as f64) / (*n as f64))
        } else {
            "—".to_string()
        };
        println!("{label:<14}  {flat:>8}  {n:>8}  {rate:>8}");
    }

    // Regime composition of the flat subset.
    let mut regime_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &flat_rows {
        *regime_counts.entry(r.regime).or_default() += 1;
    }
    println!("\n# flat-subset regime composition");
    for (k, v) in &regime_counts {
        println!("    {k:<14} {v}");
    }

    // Detrend-gate outcome decomposition. Re-run the analyzer per series
    // so we can attribute each flat row to a concrete skip reason, then
    // bucket span_ratio for the "rescue candidate" share.
    let mut gate_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut narrow_band: Vec<(String, f64, Option<f64>)> = Vec::new();
    for (series_id, full) in &series {
        let Some((train, _)) = split_series(full, TimeDelta::seconds(horizon_secs)) else {
            continue;
        };
        let train_values: Vec<f64> = train.values().filter_map(decimal_to_f64).collect();
        if train_values.len() != train.len() {
            continue;
        }
        let pipeline = rows
            .iter()
            .find(|r| &r.series_id == series_id)
            .and_then(|r| r.pipeline);
        let is_flat = pipeline
            .map(|p| p.abs() < FLAT_RETURN_EPSILON)
            .unwrap_or(false);
        if !is_flat {
            continue;
        }

        let (reason, span_ratio) = detrend_outcome(&train_values);
        *gate_counts.entry(reason).or_default() += 1;
        if reason == "span∈[0.10,0.15] (narrow-band, rescue candidate)" {
            let actual = rows
                .iter()
                .find(|r| &r.series_id == series_id)
                .and_then(|r| r.actual_return);
            narrow_band.push((series_id.clone(), span_ratio.unwrap_or(0.0), actual));
        }
    }
    println!(
        "\n# flat-subset detrend-gate outcome (n = {})",
        flat_rows.len()
    );
    let mut sorted: Vec<(&&str, &usize)> = gate_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in sorted {
        let pct = 100.0 * (*v as f64) / (flat_rows.len() as f64);
        println!("    {:<42} {:>5}  ({:>5.1}%)", k, v, pct);
    }

    if !narrow_band.is_empty() {
        println!(
            "\n# narrow-band rescue candidates (n = {}, span_ratio ∈ [0.10, 0.15])",
            narrow_band.len(),
        );
        println!(
            "{:<28}  {:>10}  {:>14}",
            "series_id", "span_ratio", "actual_return"
        );
        for (id, span, actual) in &narrow_band {
            let actual_disp = actual
                .map(|a| format!("{a:>+14.4}"))
                .unwrap_or_else(|| "             —".to_string());
            println!("{id:<28}  {span:>10.4}  {actual_disp}");
        }
    }

    // predicted_std/price distribution (zaciraci Markowitz integration: question A).
    // Verifies the assumption "flat prediction ⇒ wide band ⇒ large variance".
    // If flat group's median ratio is GREATER than non-flat's, the assumption
    // holds and predicted_std can be fed into per-token variance directly.
    // If flat ≤ non-flat, the assumption is INVERTED and zaciraci must use a
    // transform (e.g. 1/std, or a flat-detection gate orthogonal to std).
    let mut flat_ratios: Vec<f64> = Vec::new();
    let mut nonflat_ratios: Vec<f64> = Vec::new();
    for r in &rows {
        let Some(pred) = r.pipeline else { continue };
        let Some(ratio) = r.predicted_std_ratio else {
            continue;
        };
        if !ratio.is_finite() {
            continue;
        }
        if pred.abs() < FLAT_RETURN_EPSILON {
            flat_ratios.push(ratio);
        } else {
            nonflat_ratios.push(ratio);
        }
    }
    println!("\n# predicted_std / price distribution (flat vs non-flat pipeline predictions)");
    println!(
        "{:<10}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "group", "n", "p10", "p25", "p50", "p75", "p90", "mean",
    );
    print_ratio_row("flat", &mut flat_ratios);
    print_ratio_row("non-flat", &mut nonflat_ratios);

    if !flat_ratios.is_empty() && !nonflat_ratios.is_empty() {
        let flat_med = percentile(&flat_ratios, 0.50);
        let nonflat_med = percentile(&nonflat_ratios, 0.50);
        let verdict = if flat_med > nonflat_med {
            "OK: flat_median > non-flat_median (predicted_std can be fed into variance directly)"
        } else {
            "INVERTED: flat_median ≤ non-flat_median \
             (predicted_std as variance would up-weight flat predictions — \
             requires transform or orthogonal gate)"
        };
        println!(
            "\n# verdict: {verdict}\n#   flat_median = {flat_med:.6}, non-flat_median = {nonflat_med:.6}",
        );
    } else {
        println!(
            "\n# verdict: insufficient data (need both flat and non-flat predicted_std samples)"
        );
    }
}

fn percentile(sorted_or_unsorted: &[f64], q: f64) -> f64 {
    if sorted_or_unsorted.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = sorted_or_unsorted.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let idx = ((n as f64 - 1.0) * q).round() as usize;
    v[idx.min(n - 1)]
}

fn print_ratio_row(label: &str, ratios: &mut [f64]) {
    if ratios.is_empty() {
        println!(
            "{label:<10}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            0, "—", "—", "—", "—", "—", "—"
        );
        return;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ratios.len();
    let mean: f64 = ratios.iter().sum::<f64>() / n as f64;
    let p10 = percentile(ratios, 0.10);
    let p25 = percentile(ratios, 0.25);
    let p50 = percentile(ratios, 0.50);
    let p75 = percentile(ratios, 0.75);
    let p90 = percentile(ratios, 0.90);
    println!(
        "{label:<10}  {n:>5}  {p10:>10.6}  {p25:>10.6}  {p50:>10.6}  {p75:>10.6}  {p90:>10.6}  {mean:>10.6}",
    );
}
