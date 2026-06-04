//! `predict_sweep` — CLI driver around `bench::sweep::run_sweep`.
//!
//! Loads a series universe from a JSON directory, runs the predictor
//! across the Cartesian product of `(history_secs × horizon_secs ×
//! eval_dates)`, and writes the results as CSV / JSON.
//!
//! The library code lives in `bench::sweep`; this file is a thin shell:
//! clap parses arguments into [`Cli`], `From<&Cli> for SweepConfig` does
//! the conversion, then `run_sweep` does the work. Errors are surfaced
//! through `anyhow::Result` so that `?` chains cleanly in `main`.

use std::fs::File;
use std::io::BufWriter;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use bench::sweep::{
    DEFAULT_SIGNAL_THRESHOLD, SweepConfig, default_calibration_buckets, loader, output, run_sweep,
};
use chrono::NaiveDateTime;
use clap::Parser;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt};

/// Run a sweep evaluation over a directory of price series.
#[derive(Debug, Clone, Parser)]
#[command(name = "predict_sweep", version, about)]
struct Cli {
    /// Directory of JSON fixtures (one file per series).
    #[arg(long)]
    series_dir: PathBuf,

    /// History windows in seconds (comma-separated, one job per value).
    #[arg(long, value_delimiter = ',', required = true)]
    history_secs: Vec<i64>,

    /// Forecast horizons in seconds (comma-separated, one job per value).
    #[arg(long, value_delimiter = ',', required = true)]
    horizon_secs: Vec<i64>,

    /// Evaluation dates as ISO8601 timestamps without timezone, e.g.
    /// `2025-12-01T00:00:00`. Multiple values via comma.
    #[arg(long, value_delimiter = ',', required = true)]
    eval_dates: Vec<String>,

    /// Number of worker threads. Each worker owns an independent
    /// `Predictor::new(1)` so memory scales with this value.
    #[arg(long, default_value = "4")]
    workers: NonZeroUsize,

    /// Magnitude threshold for the filtered direction-accuracy metric.
    #[arg(long, default_value_t = DEFAULT_SIGNAL_THRESHOLD)]
    signal_threshold: f64,

    /// Output CSV path. Omit to skip CSV output.
    #[arg(long)]
    output_csv: Option<PathBuf>,

    /// Output JSON path (full SweepReport including aggregates).
    /// Omit to skip JSON output.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Diagnostic directory for per-job raw vectors. Unused at the
    /// moment but reserved so the CLI surface is stable.
    #[arg(long)]
    diagnostic_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    if cli.output_csv.is_none() && cli.output_json.is_none() {
        return Err(anyhow!(
            "at least one of --output-csv or --output-json must be specified"
        ));
    }

    let load = loader::load_series_dir(&cli.series_dir)
        .with_context(|| format!("loading series universe from {}", cli.series_dir.display()))?;
    if !load.errors.is_empty() {
        warn!(
            errors = load.errors.len(),
            "some fixtures failed to load and were skipped"
        );
    }
    info!(series = load.series.len(), "series universe loaded");

    let config = build_config(&cli, load.series).context("building sweep config")?;
    let report = run_sweep(config).context("running sweep")?;

    if let Some(path) = cli.output_csv.as_ref() {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        output::write_csv(&report.rows, BufWriter::new(file))
            .with_context(|| format!("writing CSV to {}", path.display()))?;
        info!(path = %path.display(), rows = report.rows.len(), "csv written");
    }
    if let Some(path) = cli.output_json.as_ref() {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        output::write_json(&report, BufWriter::new(file))
            .with_context(|| format!("writing JSON to {}", path.display()))?;
        info!(path = %path.display(), "json written");
    }

    let skipped = report.rows.iter().filter(|r| r.error.is_some()).count();
    info!(
        rows = report.rows.len(),
        skipped, "sweep complete; results emitted"
    );
    Ok(())
}

fn build_config(cli: &Cli, series_universe: Vec<bench::sweep::SeriesInput>) -> Result<SweepConfig> {
    if series_universe.is_empty() {
        return Err(anyhow!(
            "no series loaded from {}",
            cli.series_dir.display()
        ));
    }
    let eval_dates: Vec<NaiveDateTime> = cli
        .eval_dates
        .iter()
        .map(|s| {
            NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
                .with_context(|| format!("invalid eval-date '{s}'"))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(SweepConfig {
        series_universe,
        history_lens_secs: cli.history_secs.clone(),
        horizons_secs: cli.horizon_secs.clone(),
        eval_dates,
        signal_threshold: cli.signal_threshold,
        calibration_buckets: default_calibration_buckets(),
        workers: cli.workers,
        diagnostic_dir: cli.diagnostic_dir.clone(),
    })
}
