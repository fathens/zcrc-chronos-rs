#!/usr/bin/env python3
"""
JSON データ検証スクリプト。

tests/real/ 配下の JSON ファイルが健全かをチェックする:
1. DB 存在確認: データポイントが PostgreSQL の token_rates テーブルに存在するか
2. price 計算の検証: price = 10^decimals / rate が正しく計算されているか
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
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


@dataclass
class ValidationError:
    """検証エラー"""
    index: int
    timestamp: str
    error_type: str
    message: str


@dataclass
class ValidationResult:
    """検証結果"""
    file_path: Path
    total_points: int = 0
    db_checked: int = 0
    db_found: int = 0
    price_checked: int = 0
    price_valid: int = 0
    days_span: int | None = None
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return len(self.errors) == 0

    @property
    def db_ok(self) -> bool:
        """DB 確認が成功したか"""
        return not any(e.error_type in ("db_not_found", "rate_mismatch", "invalid_rate")
                       for e in self.errors)

    @property
    def price_ok(self) -> bool:
        """price 計算が成功したか"""
        return not any(e.error_type in ("price_mismatch", "invalid_number")
                       for e in self.errors)

    @property
    def days_ok(self) -> bool:
        """日数チェックが成功したか"""
        return not any(e.error_type == "invalid_days_span" for e in self.errors)


def get_db_connection():
    """DB 接続を取得"""
    return psycopg2.connect(**DB_CONFIG)


def calculate_price(rate: Decimal, decimals: int) -> Decimal:
    """rate から price を計算: price = 10^decimals / rate"""
    return Decimal(10 ** decimals) / rate


def parse_timestamp(ts_str: str) -> datetime:
    """タイムスタンプ文字列をパース"""
    # ISO 形式をパース
    return datetime.fromisoformat(ts_str)


def validate_db_existence(
    conn, base_token: str, quote_token: str, data_points: list[dict]
) -> tuple[int, int, list[ValidationError]]:
    """DB にデータが存在するか確認"""
    checked = 0
    found = 0
    errors = []

    with conn.cursor() as cur:
        for i, point in enumerate(data_points):
            ts = parse_timestamp(point["timestamp"])
            rate_str = point["rate"]

            # rate を Decimal に変換
            try:
                rate = Decimal(rate_str)
            except InvalidOperation:
                errors.append(ValidationError(
                    index=i,
                    timestamp=point["timestamp"],
                    error_type="invalid_rate",
                    message=f"Invalid rate format: {rate_str}",
                ))
                continue

            checked += 1

            # DB に存在するか確認
            cur.execute("""
                SELECT rate FROM public.token_rates
                WHERE base_token = %s
                  AND quote_token = %s
                  AND timestamp = %s
                LIMIT 1
            """, (base_token, quote_token, ts))

            row = cur.fetchone()
            if row is None:
                errors.append(ValidationError(
                    index=i,
                    timestamp=point["timestamp"],
                    error_type="db_not_found",
                    message=f"Not found in DB: {base_token}/{quote_token} at {ts}",
                ))
            else:
                db_rate = Decimal(str(row[0]))
                # rate が一致するか確認
                if db_rate != rate:
                    errors.append(ValidationError(
                        index=i,
                        timestamp=point["timestamp"],
                        error_type="rate_mismatch",
                        message=f"Rate mismatch: JSON={rate}, DB={db_rate}",
                    ))
                else:
                    found += 1

    return checked, found, errors


def validate_days_span(
    data_points: list[dict],
) -> tuple[int | None, list[ValidationError]]:
    """データが 31 日間であることを確認

    Returns:
        tuple[int | None, list[ValidationError]]: (日数, エラーリスト)
    """
    errors = []

    if len(data_points) < 2:
        errors.append(ValidationError(
            index=-1,
            timestamp="",
            error_type="invalid_days_span",
            message=f"Not enough data points to calculate days span: {len(data_points)} points",
        ))
        return None, errors

    # タイムスタンプをソートして最初と最後を取得
    timestamps = sorted(parse_timestamp(p["timestamp"]) for p in data_points)
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # 日数を計算（四捨五入）
    delta = last_ts - first_ts
    days = round(delta.total_seconds() / 86400)

    # 30〜32 日の範囲であることを確認（許容範囲）
    if not (30 <= days <= 32):
        errors.append(ValidationError(
            index=-1,
            timestamp="",
            error_type="invalid_days_span",
            message=f"Days span out of range: {days} days (expected 30-32 days, from {first_ts} to {last_ts})",
        ))

    return days, errors


def validate_price_calculation(
    data_points: list[dict], decimals: int
) -> tuple[int, int, list[ValidationError]]:
    """price の計算が正しいか確認"""
    checked = 0
    valid = 0
    errors = []

    for i, point in enumerate(data_points):
        rate_str = point["rate"]
        price_str = point["price"]

        try:
            rate = Decimal(rate_str)
            price = Decimal(price_str)
        except InvalidOperation:
            errors.append(ValidationError(
                index=i,
                timestamp=point["timestamp"],
                error_type="invalid_number",
                message=f"Invalid rate or price format",
            ))
            continue

        checked += 1

        # 期待される price を計算
        expected_price = calculate_price(rate, decimals)

        # 相対誤差を計算（浮動小数点の丸め誤差を考慮）
        if expected_price == 0:
            if price != 0:
                errors.append(ValidationError(
                    index=i,
                    timestamp=point["timestamp"],
                    error_type="price_mismatch",
                    message=f"Expected price=0, got {price}",
                ))
            else:
                valid += 1
        else:
            relative_error = abs((price - expected_price) / expected_price)
            # 1e-15 の相対誤差を許容（Decimal の精度を考慮）
            if relative_error > Decimal("1e-15"):
                errors.append(ValidationError(
                    index=i,
                    timestamp=point["timestamp"],
                    error_type="price_mismatch",
                    message=(
                        f"Price calculation error: "
                        f"expected={expected_price:.10E}, got={price:.10E}, "
                        f"relative_error={relative_error:.2E}"
                    ),
                ))
            else:
                valid += 1

    return checked, valid, errors


def validate_json_file(file_path: Path, conn) -> ValidationResult:
    """JSON ファイルを検証"""
    result = ValidationResult(file_path=file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.errors.append(ValidationError(
            index=-1,
            timestamp="",
            error_type="json_parse_error",
            message=f"Failed to parse JSON: {e}",
        ))
        return result

    # 必須フィールドの確認
    required_fields = ["base_token", "quote_token", "decimals", "data"]
    for field in required_fields:
        if field not in data:
            result.errors.append(ValidationError(
                index=-1,
                timestamp="",
                error_type="missing_field",
                message=f"Missing required field: {field}",
            ))
            return result

    base_token = data["base_token"]
    quote_token = data["quote_token"]
    decimals = data["decimals"]
    data_points = data["data"]

    result.total_points = len(data_points)

    # DB 存在確認
    checked, found, db_errors = validate_db_existence(
        conn, base_token, quote_token, data_points
    )
    result.db_checked = checked
    result.db_found = found
    result.errors.extend(db_errors)

    # price 計算の検証
    checked, valid, price_errors = validate_price_calculation(data_points, decimals)
    result.price_checked = checked
    result.price_valid = valid
    result.errors.extend(price_errors)

    # 日数チェック
    days, days_errors = validate_days_span(data_points)
    result.days_span = days
    result.errors.extend(days_errors)

    return result


def print_result(result: ValidationResult, verbose: bool = False) -> None:
    """検証結果を出力"""
    status = "OK" if result.is_ok else "NG"
    print(f"[{status}] {result.file_path}")

    # 各チェック項目の結果を個別に表示
    print(f"     データ点数: {result.total_points}")

    # DB 確認の結果
    db_status = "OK" if result.db_ok else "NG"
    print(f"     DB 確認: [{db_status}] {result.db_found}/{result.db_checked}")

    # price 検証の結果
    price_status = "OK" if result.price_ok else "NG"
    print(f"     price 検証: [{price_status}] {result.price_valid}/{result.price_checked}")

    # 日数チェックの結果
    days_status = "OK" if result.days_ok else "NG"
    days_str = f"{result.days_span} 日" if result.days_span is not None else "N/A"
    print(f"     日数チェック: [{days_status}] {days_str}")

    if not result.is_ok:
        print(f"     エラー数: {len(result.errors)}")

        if verbose:
            for error in result.errors[:10]:  # 最初の 10 件を表示
                print(f"       [{error.index}] {error.error_type}: {error.message}")
            if len(result.errors) > 10:
                print(f"       ... and {len(result.errors) - 10} more errors")


def main():
    parser = argparse.ArgumentParser(
        description="JSON データファイルを検証する"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="検証する JSON ファイル",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="エラー詳細を表示",
    )

    args = parser.parse_args()

    # DB 接続
    try:
        conn = get_db_connection()
    except psycopg2.Error as e:
        print(f"DB 接続エラー: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        results = []
        for file_path in args.files:
            if not file_path.exists():
                print(f"[NG] {file_path}")
                print(f"     ファイルが存在しません")
                results.append(ValidationResult(
                    file_path=file_path,
                    errors=[ValidationError(
                        index=-1,
                        timestamp="",
                        error_type="file_not_found",
                        message="File not found",
                    )],
                ))
                continue

            result = validate_json_file(file_path, conn)
            results.append(result)
            print_result(result, args.verbose)
            print()

        # 最終結果
        total = len(results)
        ok_count = sum(1 for r in results if r.is_ok)
        ng_count = total - ok_count

        print("=" * 60)
        print(f"結果: {ok_count}/{total} ファイル OK")

        if ng_count > 0:
            print(f"エラー: {ng_count} ファイル")
            sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
