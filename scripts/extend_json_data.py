#!/usr/bin/env python3
"""
JSON データファイルを拡張するスクリプト。

31日未満のデータを持つ JSON ファイルに対して、
DB からデータを取得して追加する。
先頭方向に追加できない場合は末尾方向に追加する。
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg2


# DB 接続情報
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "readonly",
    "password": "readonly",
}

TARGET_DAYS = 31


def get_db_connection():
    """DB 接続を取得"""
    return psycopg2.connect(**DB_CONFIG)


def parse_timestamp(ts_str: str) -> datetime:
    """タイムスタンプ文字列をパース"""
    return datetime.fromisoformat(ts_str)


def calculate_price(rate: Decimal, decimals: int) -> Decimal:
    """rate から price を計算: price = 10^decimals / rate"""
    return Decimal(10 ** decimals) / rate


def get_current_days_span(data_points: list[dict]) -> float:
    """現在のデータの日数を計算"""
    if len(data_points) < 2:
        return 0.0

    timestamps = sorted(parse_timestamp(p["timestamp"]) for p in data_points)
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    delta = last_ts - first_ts
    return delta.total_seconds() / (24 * 60 * 60)


def get_prepend_data(conn, base_token: str, quote_token: str,
                     before_timestamp: datetime, needed_days: float) -> list[tuple]:
    """
    指定タイムスタンプより前のデータを DB から取得。

    Args:
        conn: DB 接続
        base_token: ベーストークン
        quote_token: クォートトークン
        before_timestamp: この時刻より前のデータを取得
        needed_days: 必要な日数

    Returns:
        [(timestamp, rate), ...] のリスト（時系列順）
    """
    # 余裕を持って取得（必要日数 + 1日）
    start_timestamp = before_timestamp - timedelta(days=needed_days + 1)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp, rate
            FROM public.token_rates
            WHERE base_token = %s
              AND quote_token = %s
              AND timestamp >= %s
              AND timestamp < %s
            ORDER BY timestamp
        """, (base_token, quote_token, start_timestamp, before_timestamp))

        return cur.fetchall()


def get_append_data(conn, base_token: str, quote_token: str,
                    after_timestamp: datetime, needed_days: float) -> list[tuple]:
    """
    指定タイムスタンプより後のデータを DB から取得。

    Args:
        conn: DB 接続
        base_token: ベーストークン
        quote_token: クォートトークン
        after_timestamp: この時刻より後のデータを取得
        needed_days: 必要な日数

    Returns:
        [(timestamp, rate), ...] のリスト（時系列順）
    """
    # 余裕を持って取得（必要日数 + 1日）
    end_timestamp = after_timestamp + timedelta(days=needed_days + 1)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp, rate
            FROM public.token_rates
            WHERE base_token = %s
              AND quote_token = %s
              AND timestamp > %s
              AND timestamp <= %s
            ORDER BY timestamp
        """, (base_token, quote_token, after_timestamp, end_timestamp))

        return cur.fetchall()


def format_data_point(timestamp: datetime, rate: Decimal, decimals: int) -> dict:
    """データポイントを JSON 形式に変換"""
    price = calculate_price(rate, decimals)
    return {
        "timestamp": timestamp.isoformat(),
        "rate": f"{rate:.22f}",
        "price": f"{price:.49E}",
    }


def trim_to_target_days(data_points: list[dict], target_days: int,
                        direction: str) -> list[dict]:
    """
    データを目標日数に収まるようにトリミングする。

    Args:
        data_points: データポイントのリスト
        target_days: 目標日数
        direction: "先頭" または "末尾"（どちら側にデータを追加したか）

    Returns:
        トリミングされたデータポイントのリスト
    """
    if len(data_points) < 2:
        return data_points

    # タイムスタンプでソート
    sorted_points = sorted(data_points, key=lambda p: parse_timestamp(p["timestamp"]))

    # 目標日数を秒に変換
    target_seconds = target_days * 24 * 60 * 60

    if direction == "先頭":
        # 末尾を基準に、先頭からトリミング
        last_ts = parse_timestamp(sorted_points[-1]["timestamp"])
        cutoff_ts = last_ts - timedelta(seconds=target_seconds)

        trimmed = [p for p in sorted_points
                   if parse_timestamp(p["timestamp"]) >= cutoff_ts]
    else:
        # 先頭を基準に、末尾からトリミング
        first_ts = parse_timestamp(sorted_points[0]["timestamp"])
        cutoff_ts = first_ts + timedelta(seconds=target_seconds)

        trimmed = [p for p in sorted_points
                   if parse_timestamp(p["timestamp"]) <= cutoff_ts]

    return trimmed


def extend_json_file(file_path: Path, conn, target_days: int,
                     dry_run: bool = False, verbose: bool = False,
                     trim_from: str | None = None) -> bool:
    """
    JSON ファイルを拡張する。
    先頭方向に追加を試み、できなければ末尾方向に追加する。
    目標日数を超えた場合はトリミングする。

    Args:
        file_path: 対象ファイル
        conn: DB 接続
        target_days: 目標日数
        dry_run: True の場合、変更を保存しない
        verbose: True の場合、詳細出力
        trim_from: トリミング方向（"先頭" または "末尾"）。None の場合は追加方向に従う

    Returns:
        True: 成功（拡張または既に十分）
        False: 失敗
    """
    # JSON 読み込み
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[ERROR] {file_path}: ファイル読み込み失敗: {e}")
        return False

    # 必須フィールドの確認
    required_fields = ["base_token", "quote_token", "decimals", "data"]
    for field in required_fields:
        if field not in data:
            print(f"[ERROR] {file_path}: 必須フィールド '{field}' がありません")
            return False

    base_token = data["base_token"]
    quote_token = data["quote_token"]
    decimals = data["decimals"]
    data_points = data["data"]

    # 現在の日数を計算
    current_days = get_current_days_span(data_points)

    if verbose:
        print(f"  現在: {current_days:.1f} 日, {len(data_points)} データ点")

    # 既に十分な場合（許容範囲内）
    if target_days <= current_days <= target_days + 1:
        print(f"[OK] {file_path.name}: 既に {current_days:.1f} 日あります")
        return True

    # 多すぎる場合はトリミングのみ
    if current_days > target_days + 1:
        if verbose:
            print(f"  日数超過: トリミングのみ実行")

        trimmed_data = trim_to_target_days(data_points, target_days, "末尾")
        new_days = get_current_days_span(trimmed_data)

        if dry_run:
            print(f"[DRY-RUN] {file_path.name}: {current_days:.1f} 日 -> {new_days:.1f} 日 "
                  f"(トリミング: {len(data_points)} -> {len(trimmed_data)} データ点)")
            return True

        data["data"] = trimmed_data
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[TRIMMED] {file_path.name}: {current_days:.1f} 日 -> {new_days:.1f} 日 "
              f"({len(data_points)} -> {len(trimmed_data)} データ点)")
        return True

    # タイムスタンプを取得
    timestamps = [(parse_timestamp(p["timestamp"]), i) for i, p in enumerate(data_points)]
    timestamps.sort()
    first_ts = timestamps[0][0]
    last_ts = timestamps[-1][0]

    # 必要な日数を計算（余裕を持たせて取得、後でトリミング）
    needed_days = target_days - current_days + 0.5

    if verbose:
        print(f"  先頭タイムスタンプ: {first_ts}")
        print(f"  末尾タイムスタンプ: {last_ts}")
        print(f"  追加で必要な日数: {needed_days:.1f} 日")

    # 先頭方向と末尾方向のデータを取得
    prepend_data = get_prepend_data(conn, base_token, quote_token, first_ts, needed_days)
    append_data = get_append_data(conn, base_token, quote_token, last_ts, needed_days)

    if verbose:
        print(f"  先頭方向: DB から {len(prepend_data) if prepend_data else 0} 点")
        print(f"  末尾方向: DB から {len(append_data) if append_data else 0} 点")

    if not prepend_data and not append_data:
        print(f"[WARN] {file_path.name}: DB に追加データがありません（先頭・末尾とも）")
        return False

    # 両方向を試して、31日に近くなる方を選ぶ
    candidates = []

    if prepend_data:
        new_points = []
        for ts, rate in prepend_data:
            rate_decimal = Decimal(str(rate))
            new_points.append(format_data_point(ts, rate_decimal, decimals))
        extended = new_points + data_points
        extended_days = get_current_days_span(extended)
        if extended_days > target_days + 1:
            extended = trim_to_target_days(extended, target_days, "先頭")
        final_days = get_current_days_span(extended)
        candidates.append(("先頭", extended, final_days, len(new_points)))

    if append_data:
        new_points = []
        for ts, rate in append_data:
            rate_decimal = Decimal(str(rate))
            new_points.append(format_data_point(ts, rate_decimal, decimals))
        extended = data_points + new_points
        extended_days = get_current_days_span(extended)
        if extended_days > target_days + 1:
            extended = trim_to_target_days(extended, target_days, "末尾")
        final_days = get_current_days_span(extended)
        candidates.append(("末尾", extended, final_days, len(new_points)))

    # 目標日数に最も近い方を選択（現在の日数より改善している場合のみ）
    best = None
    for direction, extended, final_days, added in candidates:
        if final_days > current_days:  # 改善している
            if best is None or abs(final_days - target_days) < abs(best[2] - target_days):
                best = (direction, extended, final_days, added)

    if best is None:
        print(f"[WARN] {file_path.name}: 追加しても日数が改善しません")
        return False

    direction, extended_data, _, _ = best

    # 更新後の日数を確認
    new_days = get_current_days_span(extended_data)
    added_count = len(extended_data) - len(data_points)

    if verbose:
        print(f"  更新後: {new_days:.1f} 日, {len(extended_data)} データ点")

    if dry_run:
        print(f"[DRY-RUN] {file_path.name}: {current_days:.1f} 日 -> {new_days:.1f} 日 "
              f"({'+' if added_count >= 0 else ''}{added_count} データ点, {direction})")
        return True

    # ファイルを更新
    data["data"] = extended_data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[UPDATED] {file_path.name}: {current_days:.1f} 日 -> {new_days:.1f} 日 "
          f"({'+' if added_count >= 0 else ''}{added_count} データ点, {direction})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="JSON データファイルを拡張する（先頭または末尾にデータを追加）"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="対象の JSON ファイル",
    )
    parser.add_argument(
        "--target-days", "-t",
        type=int,
        default=TARGET_DAYS,
        help=f"目標日数 (default: {TARGET_DAYS})",
    )
    parser.add_argument(
        "--trim-from",
        choices=["先頭", "末尾"],
        help="トリミング方向（多すぎる場合のみ使用）",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="確認のみ（ファイルを更新しない）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細出力",
    )

    args = parser.parse_args()

    # DB 接続
    try:
        conn = get_db_connection()
    except psycopg2.Error as e:
        print(f"DB 接続エラー: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        success_count = 0
        fail_count = 0

        for file_path in args.files:
            if not file_path.exists():
                print(f"[ERROR] {file_path}: ファイルが存在しません")
                fail_count += 1
                continue

            if args.verbose:
                print(f"\n処理中: {file_path}")

            if extend_json_file(file_path, conn, args.target_days,
                                args.dry_run, args.verbose):
                success_count += 1
            else:
                fail_count += 1

        # 結果サマリー
        print()
        print("=" * 60)
        print(f"結果: {success_count}/{success_count + fail_count} ファイル成功")
        if fail_count > 0:
            print(f"失敗: {fail_count} ファイル")
            sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
