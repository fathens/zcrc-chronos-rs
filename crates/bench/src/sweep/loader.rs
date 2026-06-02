//! Read a `SeriesInput` universe from a directory of JSON fixtures.
//!
//! The on-disk format matches the price fixtures under `tests/real/`:
//!
//! ```json
//! {
//!   "description": "...",
//!   "base_token": "...",
//!   "quote_token": "...",
//!   "decimals": 24,
//!   "data": [
//!     {"timestamp": "2025-10-16T08:04:32.847943", "price": "1.234"}
//!   ]
//! }
//! ```
//!
//! The loader walks the directory, parses each `*.json` file, and emits
//! one [`SeriesInput`] per file, using the file stem as the series id.
//! A failure on any single file (parse error, I/O error, size cap) is
//! logged and recorded but does not abort the rest of the load — a
//! single corrupt fixture must not derail a 280-token sweep.
//!
//! Gated behind the `cli` feature so library consumers do not pull
//! `serde_json` into their dependency graph.

#![cfg(feature = "cli")]

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use bigdecimal::BigDecimal;
use chrono::NaiveDateTime;
use serde::Deserialize;
use tracing::warn;

use crate::sweep::{SeriesInput, SweepError, SweepResult};

/// Upper bound on the size of a single fixture file. 256 MiB is plenty
/// for hourly data over multiple years; anything beyond suggests a
/// runaway dump and is rejected without parsing.
pub const MAX_FILE_SIZE_BYTES: u64 = 256 * 1024 * 1024;

/// Outcome of [`load_series_dir`]. The errors vector preserves the file
/// path so the caller can surface partial-load failures in the report.
#[derive(Debug)]
pub struct LoadResult {
    pub series: Vec<SeriesInput>,
    pub errors: Vec<(PathBuf, String)>,
}

/// Read every `*.json` file in `dir` (non-recursive) and return the
/// successfully parsed series, along with a per-file error list for the
/// partial failures.
pub fn load_series_dir<P: AsRef<Path>>(dir: P) -> SweepResult<LoadResult> {
    let dir = dir.as_ref();
    let entries = fs::read_dir(dir).map_err(|e| {
        SweepError::InvalidConfig(format!("cannot read series dir {}: {e}", dir.display()))
    })?;

    let mut series = Vec::new();
    let mut errors = Vec::new();
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|s| s == "json").unwrap_or(false))
        .collect();
    // Sort so the loaded series come out in a deterministic order; this
    // matters for the sweep canonical row sort downstream.
    paths.sort();

    for path in paths {
        match load_single_file(&path) {
            Ok(input) => series.push(input),
            Err(reason) => {
                warn!(path = %path.display(), %reason, "series fixture skipped");
                errors.push((path, reason));
            }
        }
    }

    Ok(LoadResult { series, errors })
}

fn load_single_file(path: &Path) -> Result<SeriesInput, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("metadata: {e}"))?;
    if metadata.len() > MAX_FILE_SIZE_BYTES {
        return Err(format!(
            "file is {} bytes, exceeds MAX_FILE_SIZE_BYTES = {MAX_FILE_SIZE_BYTES}",
            metadata.len()
        ));
    }

    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    // BufReader avoids feeding serde_json one syscall per character.
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(metadata.len() as usize);
    reader
        .read_to_end(&mut buf)
        .map_err(|e| format!("read: {e}"))?;

    let parsed: SeriesFile = serde_json::from_slice(&buf).map_err(|e| format!("parse: {e}"))?;

    let series_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "file stem not valid UTF-8".to_string())?
        .to_string();

    if parsed.data.is_empty() {
        return Err("data array is empty".to_string());
    }

    let mut data: BTreeMap<NaiveDateTime, BigDecimal> = BTreeMap::new();
    for (i, point) in parsed.data.iter().enumerate() {
        let ts = NaiveDateTime::parse_from_str(&point.timestamp, "%Y-%m-%dT%H:%M:%S%.f")
            .map_err(|e| format!("invalid timestamp at index {i}: {e}"))?;
        let price = BigDecimal::from_str(&point.price)
            .map_err(|e| format!("invalid price at index {i}: {e}"))?;
        data.insert(ts, price);
    }

    Ok(SeriesInput { series_id, data })
}

#[derive(Debug, Deserialize)]
struct SeriesFile {
    data: Vec<DataPoint>,
}

#[derive(Debug, Deserialize)]
struct DataPoint {
    timestamp: String,
    price: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::io::Write;
    use tempdir_helpers::TempDir;

    /// Tiny tempdir helper to keep the test self-contained without
    /// pulling in another dev-dependency.
    mod tempdir_helpers {
        use std::env;
        use std::fs;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        pub struct TempDir {
            path: PathBuf,
        }

        impl TempDir {
            pub fn new(prefix: &str) -> Self {
                let id = COUNTER.fetch_add(1, Ordering::Relaxed);
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let path = env::temp_dir().join(format!("{prefix}-{id}-{nanos}"));
                fs::create_dir_all(&path).expect("create tempdir");
                Self { path }
            }
            pub fn path(&self) -> &std::path::Path {
                &self.path
            }
        }

        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }

    fn write_fixture(dir: &Path, name: &str, payload: &str) {
        let mut file = File::create(dir.join(name)).unwrap();
        file.write_all(payload.as_bytes()).unwrap();
    }

    #[test]
    fn loads_valid_fixture_and_uses_stem_as_id() {
        let tmp = TempDir::new("sweep-loader-valid");
        let payload = r#"{
            "data": [
                {"timestamp": "2024-01-01T00:00:00", "price": "100.5"},
                {"timestamp": "2024-01-01T01:00:00", "price": "101.0"}
            ]
        }"#;
        write_fixture(tmp.path(), "BTC.json", payload);
        let result = load_series_dir(tmp.path()).unwrap();
        assert_eq!(result.series.len(), 1);
        assert!(result.errors.is_empty());
        let series = &result.series[0];
        assert_eq!(series.series_id, "BTC");
        assert_eq!(series.data.len(), 2);
    }

    #[test]
    fn skips_unparseable_file_but_keeps_valid_ones() {
        let tmp = TempDir::new("sweep-loader-partial");
        write_fixture(tmp.path(), "broken.json", "{ not json");
        let good = r#"{
            "data": [{"timestamp": "2024-01-01T00:00:00", "price": "1.0"}]
        }"#;
        write_fixture(tmp.path(), "good.json", good);
        let result = load_series_dir(tmp.path()).unwrap();
        assert_eq!(result.series.len(), 1);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.series[0].series_id, "good");
        let (path, reason) = &result.errors[0];
        assert!(path.ends_with("broken.json"));
        assert!(reason.contains("parse"));
    }

    #[test]
    fn rejects_empty_data_array() {
        let tmp = TempDir::new("sweep-loader-empty");
        write_fixture(tmp.path(), "empty.json", r#"{"data": []}"#);
        let result = load_series_dir(tmp.path()).unwrap();
        assert!(result.series.is_empty());
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].1.contains("empty"));
    }

    #[test]
    fn invalid_timestamp_is_reported_with_index() {
        let tmp = TempDir::new("sweep-loader-bad-ts");
        let payload = r#"{
            "data": [
                {"timestamp": "not-a-date", "price": "1.0"}
            ]
        }"#;
        write_fixture(tmp.path(), "bad.json", payload);
        let result = load_series_dir(tmp.path()).unwrap();
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].1.contains("timestamp at index 0"));
    }

    #[test]
    fn ignores_non_json_files() {
        let tmp = TempDir::new("sweep-loader-nonjson");
        write_fixture(tmp.path(), "README.md", "ignored");
        let payload = r#"{
            "data": [{"timestamp": "2024-01-01T00:00:00", "price": "1.0"}]
        }"#;
        write_fixture(tmp.path(), "ok.json", payload);
        let result = load_series_dir(tmp.path()).unwrap();
        assert_eq!(result.series.len(), 1);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn returns_error_when_dir_missing() {
        let res = load_series_dir("/non/existent/path/should/never/exist");
        assert!(matches!(res, Err(SweepError::InvalidConfig(_))));
    }
}
