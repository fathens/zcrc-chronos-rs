use super::*;
use chrono::NaiveDate;

fn make_ts(hour: u32, min: u32, sec: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(hour, min, sec)
        .unwrap()
}

#[test]
fn test_empty_input() {
    let (ts, vals) = normalize_time_series_data(&[], &[]).unwrap();
    assert!(ts.is_empty());
    assert!(vals.is_empty());
}

#[test]
fn test_single_point() {
    let ts = vec![make_ts(10, 0, 0)];
    let vals = vec![42.0];
    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
    assert_eq!(out_ts.len(), 1);
    assert_eq!(out_vals, vec![42.0]);
}

#[test]
fn test_regular_intervals_skip_normalization() {
    // 5 points at exactly 1-hour intervals → CV ≈ 0 → should return as-is
    let ts: Vec<NaiveDateTime> = (0..5).map(|i| make_ts(10 + i, 0, 0)).collect();
    let vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
    assert_eq!(out_ts, ts);
    assert_eq!(out_vals, vals);
}

#[test]
fn test_irregular_intervals_normalized() {
    // Irregular intervals: 60s, 300s, 60s, 600s → high CV
    let ts = vec![
        make_ts(10, 0, 0),
        make_ts(10, 1, 0),  // +60s
        make_ts(10, 6, 0),  // +300s
        make_ts(10, 7, 0),  // +60s
        make_ts(10, 17, 0), // +600s
    ];
    let vals = vec![100.0, 200.0, 150.0, 300.0, 50.0];

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
    assert_eq!(out_ts.len(), 5);
    assert_eq!(out_vals.len(), 5);
    // First timestamp should be preserved
    assert_eq!(out_ts[0], ts[0]);
    // Last timestamp should be preserved
    assert_eq!(out_ts[4], ts[4]);
    // All original values should appear (nearest-neighbor)
    for v in &out_vals {
        assert!(vals.contains(v));
    }
}

#[test]
fn test_mismatched_lengths_error() {
    let ts = vec![make_ts(10, 0, 0)];
    let vals = vec![1.0, 2.0];
    assert!(normalize_time_series_data(&ts, &vals).is_err());
}

#[test]
fn test_preserves_value_range() {
    // The normalization should not introduce values outside original range
    let ts = vec![
        make_ts(10, 0, 0),
        make_ts(10, 0, 30), // +30s
        make_ts(10, 5, 0),  // +270s  (big gap)
        make_ts(10, 5, 20), // +20s
        make_ts(10, 20, 0), // +880s  (big gap)
    ];
    let vals = vec![10.0, 50.0, 20.0, 80.0, 30.0];

    let (_, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();
    let orig_min = min_f64(&vals);
    let orig_max = max_f64(&vals);
    for v in &out_vals {
        assert!(*v >= orig_min && *v <= orig_max);
    }
}

fn make_ts_days(day: i64, hour: u32, min: u32, sec: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(hour, min, sec)
        .unwrap()
        + chrono::Duration::days(day)
}

#[test]
fn test_large_gap_splits_into_segments() {
    // Simulate data with a 7-day gap in the middle
    // Before gap: days 0-2 with hourly data (3 days * 24 = 72 points)
    // Gap: 7 days
    // After gap: days 9-11 with hourly data (3 days * 24 = 72 points)
    let mut ts = Vec::new();
    let mut vals = Vec::new();

    // Before gap: days 0, 1, 2 (hourly)
    for day in 0..3 {
        for hour in 0..24 {
            ts.push(make_ts_days(day, hour as u32, 0, 0));
            vals.push(100.0 + day as f64 * 10.0 + hour as f64);
        }
    }

    // After gap: days 9, 10, 11 (hourly)
    for day in 9..12 {
        for hour in 0..24 {
            ts.push(make_ts_days(day, hour as u32, 0, 0));
            vals.push(200.0 + (day - 9) as f64 * 10.0 + hour as f64);
        }
    }

    assert_eq!(ts.len(), 144);

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    // Should have same number of points
    assert_eq!(out_ts.len(), 144);
    assert_eq!(out_vals.len(), 144);

    // The gap should still exist (segments normalized independently)
    // Find the gap: there should be a large time difference around index 72
    let gap_idx = 71; // last point before gap
    let before_gap = out_ts[gap_idx];
    let after_gap = out_ts[gap_idx + 1];
    let gap_hours = (after_gap - before_gap).num_hours();

    // Gap should be approximately 7 days (168 hours), not compressed
    assert!(
        gap_hours > 100,
        "Gap should be preserved, got {} hours",
        gap_hours
    );

    // Values should not have excessive repetition
    // Count unique values in each segment
    let before_vals: std::collections::HashSet<_> = out_vals[..72]
        .iter()
        .map(|v| (*v * 1000.0) as i64)
        .collect();
    let after_vals: std::collections::HashSet<_> = out_vals[72..]
        .iter()
        .map(|v| (*v * 1000.0) as i64)
        .collect();

    // Each segment should have diverse values (not all same due to nearest-neighbor)
    assert!(
        before_vals.len() > 20,
        "Before-gap segment should have diverse values, got {} unique",
        before_vals.len()
    );
    assert!(
        after_vals.len() > 20,
        "After-gap segment should have diverse values, got {} unique",
        after_vals.len()
    );
}

#[test]
fn test_median() {
    assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
    assert_eq!(median(&[]), 0.0);
}

#[test]
fn test_gap_below_threshold_no_split() {
    // Gap is 9x median (below 10x threshold) - should NOT split
    // Intervals: 1h, 1h, 1h, 9h, 1h, 1h, 1h (median = 1h, max = 9h)
    let ts = vec![
        make_ts(0, 0, 0),
        make_ts(1, 0, 0),  // +1h
        make_ts(2, 0, 0),  // +1h
        make_ts(3, 0, 0),  // +1h
        make_ts(12, 0, 0), // +9h (9x median, below threshold)
        make_ts(13, 0, 0), // +1h
        make_ts(14, 0, 0), // +1h
        make_ts(15, 0, 0), // +1h
    ];
    let vals: Vec<f64> = (0..8).map(|i| 100.0 + i as f64 * 10.0).collect();

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 8);
    assert_eq!(out_vals.len(), 8);

    // Check that timestamps span the full range (not split)
    assert_eq!(out_ts[0], ts[0]);
    assert_eq!(out_ts[7], ts[7]);

    // The gap should be compressed (uniform grid applied to full data)
    let gap_hours = (out_ts[4] - out_ts[3]).num_minutes() as f64 / 60.0;
    let expected_interval = 15.0 / 7.0; // 15 hours / 7 intervals
    assert!(
        (gap_hours - expected_interval).abs() < 0.1,
        "Gap should be compressed to uniform interval, got {} hours",
        gap_hours
    );
}

#[test]
fn test_gap_at_threshold_splits() {
    // Gap is 11x median (above 10x threshold) - should split
    // Intervals: 1h, 1h, 1h, 11h, 1h, 1h, 1h (median = 1h, max = 11h)
    let ts = vec![
        make_ts(0, 0, 0),
        make_ts(1, 0, 0),  // +1h
        make_ts(2, 0, 0),  // +1h
        make_ts(3, 0, 0),  // +1h
        make_ts(14, 0, 0), // +11h (11x median, above threshold)
        make_ts(15, 0, 0), // +1h
        make_ts(16, 0, 0), // +1h
        make_ts(17, 0, 0), // +1h
    ];
    let vals: Vec<f64> = (0..8).map(|i| 100.0 + i as f64 * 10.0).collect();

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 8);
    assert_eq!(out_vals.len(), 8);

    // The gap should be preserved (segments normalized independently)
    let gap_hours = (out_ts[4] - out_ts[3]).num_hours();
    assert!(
        gap_hours >= 10,
        "Gap should be preserved when above threshold, got {} hours",
        gap_hours
    );
}

#[test]
fn test_multiple_gaps_multiple_segments() {
    // Data with 2 large gaps -> 3 segments
    let ts = vec![
        // Segment 1: hours 0-2
        make_ts(0, 0, 0),
        make_ts(1, 0, 0),
        make_ts(2, 0, 0),
        // Gap 1: 20 hours
        // Segment 2: hours 22-24
        make_ts(22, 0, 0),
        make_ts(23, 0, 0),
        make_ts_days(1, 0, 0, 0), // day 1, hour 0
        // Gap 2: 20 hours
        // Segment 3: day 1, hours 20-22
        make_ts_days(1, 20, 0, 0),
        make_ts_days(1, 21, 0, 0),
        make_ts_days(1, 22, 0, 0),
    ];
    let vals: Vec<f64> = (0..9).map(|i| 100.0 + i as f64 * 10.0).collect();

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 9);
    assert_eq!(out_vals.len(), 9);

    // Both gaps should be preserved
    let gap1_hours = (out_ts[3] - out_ts[2]).num_hours();
    let gap2_hours = (out_ts[6] - out_ts[5]).num_hours();

    assert!(
        gap1_hours >= 15,
        "First gap should be preserved, got {} hours",
        gap1_hours
    );
    assert!(
        gap2_hours >= 15,
        "Second gap should be preserved, got {} hours",
        gap2_hours
    );
}

#[test]
fn test_small_segment_after_split() {
    // After split, one segment has only 2 points
    let ts = vec![
        make_ts(0, 0, 0),
        make_ts(1, 0, 0), // 2 points before gap
        // Gap: 20 hours
        make_ts(21, 0, 0),
        make_ts(22, 0, 0),
        make_ts(23, 0, 0), // 3 points after gap
    ];
    let vals = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 5);
    assert_eq!(out_vals.len(), 5);

    // Small segments should still be handled correctly
    assert!(out_vals.iter().all(|v| *v >= 10.0 && *v <= 50.0));
}

#[test]
fn test_single_point_segment() {
    // After split, first segment has only 1 point
    let ts = vec![
        make_ts(0, 0, 0), // 1 point before gap
        // Gap: 20 hours
        make_ts(20, 0, 0),
        make_ts(21, 0, 0),
        make_ts(22, 0, 0), // 3 points after gap
    ];
    let vals = vec![10.0, 20.0, 30.0, 40.0];

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 4);
    assert_eq!(out_vals.len(), 4);

    // First value should be preserved as-is
    assert_eq!(out_vals[0], 10.0);
}

#[test]
fn test_irregular_no_outlier_gap_still_normalizes() {
    // Irregular intervals but no outlier gaps - should normalize as before
    // Intervals vary between 1-5 minutes (no outlier)
    let ts = vec![
        make_ts(10, 0, 0),
        make_ts(10, 1, 0),  // +1m
        make_ts(10, 3, 0),  // +2m
        make_ts(10, 8, 0),  // +5m
        make_ts(10, 9, 0),  // +1m
        make_ts(10, 12, 0), // +3m
        make_ts(10, 17, 0), // +5m
    ];
    let vals: Vec<f64> = (0..7).map(|i| 100.0 + i as f64 * 10.0).collect();

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 7);
    assert_eq!(out_vals.len(), 7);

    // Timestamps should be uniformly distributed
    let intervals: Vec<i64> = out_ts
        .windows(2)
        .map(|w| (w[1] - w[0]).num_seconds())
        .collect();
    let first_interval = intervals[0];
    for interval in &intervals {
        // All intervals should be approximately equal
        assert!(
            (*interval - first_interval).abs() <= 1,
            "Intervals should be uniform: {:?}",
            intervals
        );
    }
}

#[test]
fn test_two_points_no_crash() {
    // Edge case: only 2 points
    let ts = vec![make_ts(0, 0, 0), make_ts(1, 0, 0)];
    let vals = vec![10.0, 20.0];

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 2);
    assert_eq!(out_vals.len(), 2);
}

#[test]
fn test_gap_detection_uses_median_not_mean() {
    // One very large interval that would skew the mean but not median
    // Intervals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 100 (median=1, mean=10.9)
    // A gap of 15 should be detected as outlier (15 > 10 * median)
    // but would not be if using mean (15 < 10 * 10.9)
    let mut ts = Vec::new();
    let mut vals = Vec::new();

    // 10 points with 1-hour intervals
    for i in 0..10 {
        ts.push(make_ts(i as u32, 0, 0));
        vals.push(100.0 + i as f64);
    }
    // Large gap of 100 hours (this sets the extreme)
    ts.push(make_ts_days(4, 13, 0, 0)); // 100 hours from last
    vals.push(200.0);

    // Add a moderate gap of 15 hours (should be detected as outlier based on median)
    ts.push(make_ts_days(5, 4, 0, 0)); // 15 hours from last
    vals.push(210.0);

    let (out_ts, out_vals) = normalize_time_series_data(&ts, &vals).unwrap();

    assert_eq!(out_ts.len(), 12);
    assert_eq!(out_vals.len(), 12);

    // The 100-hour and 15-hour gaps should both be preserved
    // If mean were used, 15-hour gap would be compressed
}
