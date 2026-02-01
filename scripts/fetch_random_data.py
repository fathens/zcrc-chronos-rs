#!/usr/bin/env python3
"""
PostgreSQL からランダムにデータを取得するスクリプト。

DB から利用可能なトークンペアをランダムに選択し、
指定された数のテストデータファイルを生成する。
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import psycopg2


# DB 接続情報
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "readonly",
    "password": "readonly",
}

WINDOW_DAYS = 31


@dataclass
class TokenPairInfo:
    """トークンペア情報"""
    base_token: str
    quote_token: str
    decimals: int
    min_timestamp: datetime
    max_timestamp: datetime


@dataclass
class SelectedData:
    """選択されたデータ情報"""
    base_token: str
    quote_token: str
    decimals: int
    start_date: datetime
    end_date: datetime


def get_db_connection():
    """DB 接続を取得"""
    return psycopg2.connect(**DB_CONFIG)


def get_token_pairs_with_info(conn) -> list[TokenPairInfo]:
    """利用可能なトークンペアとその情報を取得"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                base_token,
                quote_token,
                MAX(decimals) as decimals,
                MIN(timestamp) as min_ts,
                MAX(timestamp) as max_ts
            FROM public.token_rates
            WHERE decimals IS NOT NULL
            GROUP BY base_token, quote_token
            HAVING COUNT(*) >= 100
            ORDER BY base_token, quote_token
        """)
        return [
            TokenPairInfo(
                base_token=row[0],
                quote_token=row[1],
                decimals=row[2],
                min_timestamp=row[3],
                max_timestamp=row[4],
            )
            for row in cur.fetchall()
        ]


def get_token_data(conn, base_token: str, quote_token: str,
                   start_date: datetime, end_date: datetime) -> list[dict]:
    """指定期間のトークンデータを取得"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp, rate
            FROM public.token_rates
            WHERE base_token = %s AND quote_token = %s
              AND timestamp >= %s AND timestamp < %s
            ORDER BY timestamp
        """, (base_token, quote_token, start_date, end_date))

        rows = cur.fetchall()
        return [{"timestamp": row[0], "rate": Decimal(str(row[1]))} for row in rows]


def calculate_price(rate: Decimal, decimals: int) -> Decimal:
    """rate から price を計算"""
    return Decimal(1) / rate * Decimal(10 ** decimals)


def select_random_data(token_pairs: list[TokenPairInfo],
                       already_selected: list[SelectedData],
                       rng: random.Random) -> Optional[SelectedData]:
    """
    ランダムにトークンペアと期間を選択。
    重複を避けて選択する。
    """
    if not token_pairs:
        return None

    # 有効なトークンペアをフィルタ（31日以上のデータがあるもの）
    valid_pairs = [
        p for p in token_pairs
        if (p.max_timestamp - p.min_timestamp).days >= WINDOW_DAYS
    ]

    if not valid_pairs:
        return None

    max_attempts = 100
    for _ in range(max_attempts):
        # ランダムにトークンペアを選択
        pair = rng.choice(valid_pairs)

        # 有効期間内からランダムに開始日を選択
        available_days = (pair.max_timestamp - pair.min_timestamp).days - WINDOW_DAYS
        if available_days <= 0:
            continue

        random_offset = rng.randint(0, available_days)
        start_date = pair.min_timestamp + timedelta(days=random_offset)
        # 日付の開始時刻にする
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=WINDOW_DAYS)

        selected = SelectedData(
            base_token=pair.base_token,
            quote_token=pair.quote_token,
            decimals=pair.decimals,
            start_date=start_date,
            end_date=end_date,
        )

        # 重複チェック
        if not is_duplicate(selected, already_selected):
            return selected

    return None


def is_duplicate(selected: SelectedData, already_selected: list[SelectedData]) -> bool:
    """既存の選択と重複しているかチェック"""
    for existing in already_selected:
        # 同じトークンペア・同じ開始日なら重複
        if (existing.base_token == selected.base_token and
            existing.quote_token == selected.quote_token and
            existing.start_date.date() == selected.start_date.date()):
            return True
    return False


def format_output_data(data: list[dict], selected: SelectedData) -> dict:
    """出力用 JSON データを生成"""
    decimals = selected.decimals

    data_list = []
    for row in data:
        rate = row["rate"]
        price = calculate_price(rate, decimals)
        data_list.append({
            "timestamp": row["timestamp"].isoformat(),
            "rate": f"{rate:.22f}",
            "price": f"{price:.49E}",
        })

    start_str = selected.start_date.strftime("%Y-%m-%d")
    end_str = selected.end_date.strftime("%Y-%m-%d")

    description = (
        f"ランダム取得データ。"
        f"期間: {start_str} 〜 {end_str} ({WINDOW_DAYS}日間), "
        f"データ点数: {len(data_list)}"
    )

    return {
        "description": description,
        "base_token": selected.base_token,
        "quote_token": selected.quote_token,
        "decimals": decimals,
        "data": data_list,
    }


def save_data(output_data: dict, output_dir: Path, file_number: int,
              dry_run: bool = False) -> Optional[Path]:
    """データを保存"""
    filename = f"random-{file_number:02d}.json"
    filepath = output_dir / filename

    if dry_run:
        print(f"[dry-run] Would create: {filepath}")
        print(f"  Description: {output_data['description']}")
        print(f"  Token pair: {output_data['base_token']}/{output_data['quote_token']}")
        return filepath

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL からランダムにデータを取得"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        required=True,
        help="取得するデータファイル数",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("tests/real/.tmp"),
        help="出力先ディレクトリ (default: tests/real/.tmp)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="ランダムシード（再現性のため）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際にファイル出力しない",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細出力",
    )

    args = parser.parse_args()

    if args.count <= 0:
        print("エラー: --count は正の整数を指定してください", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output.resolve()

    if not output_dir.exists():
        if args.dry_run:
            print(f"[dry-run] Would create directory: {output_dir}")
        else:
            output_dir.mkdir(parents=True)

    # ランダムジェネレータを初期化
    rng = random.Random(args.seed)

    if args.seed is not None:
        print(f"ランダムシード: {args.seed}")

    print(f"取得ファイル数: {args.count}")
    print(f"出力先: {output_dir}")

    try:
        conn = get_db_connection()
    except psycopg2.Error as e:
        print(f"DB 接続エラー: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        token_pairs = get_token_pairs_with_info(conn)
        if args.verbose:
            print(f"利用可能なトークンペア: {len(token_pairs)} 組")

        selected_list: list[SelectedData] = []
        saved_count = 0

        for i in range(args.count):
            selected = select_random_data(token_pairs, selected_list, rng)

            if selected is None:
                print(f"警告: これ以上ユニークなデータを選択できません (取得済み: {saved_count})", file=sys.stderr)
                break

            selected_list.append(selected)

            if args.verbose:
                pair_name = f"{selected.base_token}/{selected.quote_token}"
                start_str = selected.start_date.strftime("%Y-%m-%d")
                print(f"選択 {i + 1}: {pair_name} ({start_str} 〜)")

            # データ取得
            data = get_token_data(
                conn,
                selected.base_token,
                selected.quote_token,
                selected.start_date,
                selected.end_date,
            )

            if len(data) < 100:
                if args.verbose:
                    print(f"  スキップ: データ点数が少なすぎます ({len(data)} 点)")
                continue

            # 出力データ生成
            output_data = format_output_data(data, selected)

            # 保存
            file_number = saved_count + 1
            filepath = save_data(output_data, output_dir, file_number, args.dry_run)

            if filepath:
                saved_count += 1
                if not args.dry_run:
                    print(f"保存: {filepath.name} ({len(data)} 点)")

        print(f"\n完了: {saved_count} ファイル生成")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
