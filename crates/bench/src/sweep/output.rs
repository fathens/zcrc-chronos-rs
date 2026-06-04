//! CSV and JSON serialisation for sweep results.
//!
//! Two outputs are produced:
//! - **CSV** (`write_csv`): one row per [`SweepRow`], optimised for spreadsheets
//!   and pandas. Missing or non-finite numeric fields render as empty cells.
//! - **JSON** (`write_json`): the full [`SweepReport`], including the
//!   per-regime, per-series, and cross-sectional aggregates produced by
//!   `crate::sweep::aggregate`.
//!
//! Both writers reject NaN / ±Inf values at the boundary: `run_one`
//! already filters non-finite metric values into `None`, but the
//! [`serialize_finite_opt`] helper guards against drift in case future
//! callers emit `Some(NaN)` accidentally.
//!
//! Available behind the `cli` feature (along with `csv` and `serde_json`).

#![cfg(feature = "cli")]

use std::io::Write;

use serde::Serializer;

use crate::sweep::{SweepError, SweepReport, SweepRow};

/// Write every row in `rows` as a CSV record with a header row.
pub fn write_csv<W: Write>(rows: &[SweepRow], w: W) -> Result<(), SweepError> {
    let mut writer = csv::WriterBuilder::new().has_headers(true).from_writer(w);
    for row in rows {
        writer.serialize(row).map_err(map_csv_error)?;
    }
    writer.flush().map_err(map_io_error)?;
    Ok(())
}

/// Write `report` as a JSON document. Indented for human inspection.
pub fn write_json<W: Write>(report: &SweepReport, w: W) -> Result<(), SweepError> {
    serde_json::to_writer_pretty(w, report).map_err(map_json_error)?;
    Ok(())
}

/// Custom serializer for `Option<f64>` fields. `None` is serialised as an
/// empty/null value; `Some(v)` is serialised as `v` only when `v` is
/// finite, otherwise as `None`. This is defensive — the runner is the
/// authoritative source of `None`-on-non-finite — but it guarantees that
/// no `NaN` or `Inf` escapes through any future SweepRow producer that
/// forgets the filter.
pub fn serialize_finite_opt<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(v) if v.is_finite() => serializer.serialize_some(v),
        _ => serializer.serialize_none(),
    }
}

fn map_csv_error(err: csv::Error) -> SweepError {
    // csv errors come in two flavours: I/O and serde-driven. Both map onto
    // InvalidConfig so the sweep driver can record the failure on a row
    // without coupling SweepError to the csv crate's enum.
    SweepError::InvalidConfig(format!("csv: {err}"))
}

fn map_json_error(err: serde_json::Error) -> SweepError {
    SweepError::InvalidConfig(format!("json: {err}"))
}

fn map_io_error(err: std::io::Error) -> SweepError {
    SweepError::InvalidConfig(format!("io: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::{NaiveDate, NaiveDateTime};

    use crate::sweep::{SweepReport, SweepRow};

    fn ts() -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
    }

    fn populated_row() -> SweepRow {
        SweepRow {
            series_id: "BTC".into(),
            eval_date: ts(),
            history_secs: 3600 * 24 * 30,
            horizon_secs: 3600 * 24,
            model_name: Some("ensemble".into()),
            strategy_name: Some("default".into()),
            regime: Some("Trending".into()),
            train_last_ts: Some(ts()),
            train_last_value: Some(100.0),
            pred_first_ts: Some(ts()),
            actual_first_ts: Some(ts()),
            aligned_horizon_steps: Some(24),
            max_alignment_gap_secs: Some(60),
            mae: Some(0.5),
            rmse: Some(0.7),
            mape: Some(1.2),
            mase: Some(0.9),
            wape: Some(0.4),
            dir_acc: Some(0.55),
            dir_acc_filtered: Some(0.6),
            filtered_count: Some(12),
            per_row_ic: Some(0.1),
            calibration_residual: Some(0.01),
            predicted_std_first: Some(0.5),
            predicted_std_last: Some(2.5),
            final_pred_return: Some(0.02),
            final_actual_return: Some(0.018),
            processing_time_secs: Some(1.5),
            error: None,
            diagnostic_path: None,
        }
    }

    #[test]
    fn csv_emits_header_and_row() {
        let rows = vec![populated_row()];
        let mut buf = Vec::new();
        write_csv(&rows, &mut buf).expect("csv write succeeds");
        let text = String::from_utf8(buf).expect("utf8 csv");
        let mut lines = text.lines();
        let header = lines.next().expect("header line");
        assert!(header.contains("series_id"));
        assert!(header.contains("dir_acc"));
        let data = lines.next().expect("data line");
        assert!(data.starts_with("BTC,"));
        // No further rows.
        assert!(lines.next().is_none());
    }

    #[test]
    fn csv_renders_none_metric_as_empty_cell() {
        let mut row = populated_row();
        row.mae = None;
        let rows = vec![row];
        let mut buf = Vec::new();
        write_csv(&rows, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let data_line = text.lines().nth(1).unwrap();
        // Locate mae column index via the header.
        let header = text.lines().next().unwrap();
        let mae_col = header
            .split(',')
            .position(|h| h == "mae")
            .expect("mae column present");
        let cells: Vec<&str> = data_line.split(',').collect();
        assert_eq!(cells[mae_col], "");
    }

    #[test]
    fn json_round_trips_full_report() {
        let report = SweepReport {
            rows: vec![populated_row()],
            regime_summary: vec![],
            series_summary: vec![],
            horizon_summary: vec![],
            cross_sectional_ic_by_date: Default::default(),
            cross_sectional_ic_n: Default::default(),
        };
        let mut buf = Vec::new();
        write_json(&report, &mut buf).unwrap();
        let parsed: SweepReport = serde_json::from_slice(&buf).expect("json round-trip");
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].series_id, "BTC");
        assert_eq!(parsed.rows[0].mae, Some(0.5));
    }

    #[test]
    fn serialize_finite_opt_drops_nan_and_inf() {
        // Use a transparent newtype-style serializer to exercise the
        // helper without touching SweepRow.
        #[derive(serde::Serialize)]
        struct Holder {
            #[serde(serialize_with = "serialize_finite_opt")]
            v: Option<f64>,
        }
        let nan_value = serde_json::to_value(Holder { v: Some(f64::NAN) }).unwrap();
        assert_eq!(nan_value["v"], serde_json::Value::Null);
        let inf_value = serde_json::to_value(Holder {
            v: Some(f64::INFINITY),
        })
        .unwrap();
        assert_eq!(inf_value["v"], serde_json::Value::Null);
        let normal = serde_json::to_value(Holder { v: Some(1.5) }).unwrap();
        assert_eq!(normal["v"], serde_json::json!(1.5));
        let none = serde_json::to_value(Holder { v: None }).unwrap();
        assert_eq!(none["v"], serde_json::Value::Null);
    }
}
