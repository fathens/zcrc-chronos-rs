#!/usr/bin/env python3
"""
PostgreSQL からパターンデータを取得するスクリプト。

既存の tests/real/ ディレクトリにあるパターンにマッチするデータを DB から探し、
新しいテストデータとして出力する。
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats


# DB 接続情報
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "readonly",
    "password": "readonly",
}

# パターン定義
PATTERNS = [
    "uptrend",
    "downtrend",
    "high_volatility",
    "low_volatility",
    "range",
    "spike_up",
    "spike_down",
    "v_recovery",
    "double_bottom",
    "gradual_change",
]

# パターンの日本語説明
PATTERN_DESCRIPTIONS = {
    "uptrend": "上昇トレンド: 期間を通じて価格が継続的に上昇するパターン",
    "downtrend": "下降トレンド: 期間を通じて価格が継続的に下落するパターン",
    "high_volatility": "高ボラティリティ: 価格が大きく変動するパターン",
    "low_volatility": "低ボラティリティ: 価格が安定して推移するパターン",
    "range": "レンジ相場: 一定の価格帯を横ばいで推移するパターン",
    "spike_up": "急騰: 短期間で価格が急激に上昇するパターン",
    "spike_down": "急落: 短期間で価格が急激に下落するパターン",
    "v_recovery": "V字回復: 下落後に急激に回復するパターン",
    "double_bottom": "ダブルボトム: 2つの局所的安値を形成するパターン",
    "gradual_change": "緩やかな変化: 穏やかに価格が変動するパターン",
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
class PatternMatch:
    """パターンマッチ結果"""
    pattern: str
    token_pair: str
    decimals: int
    start_date: datetime
    end_date: datetime
    data_points: int
    price_change: float
    confidence: float
    metrics: dict


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
                   start_date: datetime, end_date: datetime) -> pd.DataFrame:
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
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "rate"])
        df["rate"] = df["rate"].apply(lambda x: Decimal(str(x)))
        return df


def calculate_price(rate: Decimal, decimals: int) -> Decimal:
    """rate から price を計算"""
    return Decimal(1) / rate * Decimal(10 ** decimals)


def analyze_pattern(df: pd.DataFrame, decimals: int) -> Optional[tuple[str, float, dict]]:
    """
    データフレームからパターンを検出。
    戻り値: (パターン名, 信頼度, メトリクス) または None
    """
    if len(df) < 100:
        return None

    prices = df["rate"].apply(lambda r: float(calculate_price(r, decimals))).values

    # 基本統計量
    price_change = (prices[-1] - prices[0]) / prices[0] * 100
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    cv = std_price / mean_price if mean_price != 0 else 0  # 変動係数

    # 線形回帰
    x = np.arange(len(prices))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
    normalized_slope = slope / mean_price * len(prices) * 100  # 正規化した傾き

    # Z-score
    z_scores = (prices - mean_price) / std_price if std_price > 0 else np.zeros_like(prices)
    max_z = np.max(z_scores)
    min_z = np.min(z_scores)

    # 前半・後半の分析
    mid = len(prices) // 2
    first_half_change = (prices[mid] - prices[0]) / prices[0] * 100 if prices[0] != 0 else 0
    second_half_change = (prices[-1] - prices[mid]) / prices[mid] * 100 if prices[mid] != 0 else 0

    # 局所的な極値を検出
    local_minima = find_local_minima(prices)

    metrics = {
        "price_change": price_change,
        "cv": cv,
        "r_squared": r_value ** 2,
        "normalized_slope": normalized_slope,
        "max_z_score": max_z,
        "min_z_score": min_z,
        "first_half_change": first_half_change,
        "second_half_change": second_half_change,
        "local_minima_count": len(local_minima),
    }

    # パターン判定（優先順位順）
    detected = detect_pattern(metrics, prices, local_minima)
    if detected:
        return detected[0], detected[1], metrics

    return None


def find_local_minima(prices: np.ndarray, window: int = 50) -> list[int]:
    """局所的な極小値のインデックスを検出"""
    minima = []
    for i in range(window, len(prices) - window):
        local_window = prices[i - window:i + window + 1]
        if prices[i] == np.min(local_window):
            # 既存の極小値と十分離れているか確認
            if not minima or i - minima[-1] > window * 2:
                minima.append(i)
    return minima


def detect_pattern(metrics: dict, prices: np.ndarray,
                   local_minima: list[int]) -> Optional[tuple[str, float]]:
    """メトリクスからパターンを判定"""
    price_change = metrics["price_change"]
    cv = metrics["cv"]
    r_squared = metrics["r_squared"]
    max_z = metrics["max_z_score"]
    min_z = metrics["min_z_score"]
    first_half = metrics["first_half_change"]
    second_half = metrics["second_half_change"]

    # spike_up: 急騰
    if max_z > 3 and price_change > 30:
        return ("spike_up", min(0.9, max_z / 5))

    # spike_down: 急落
    if min_z < -3 and price_change < -30:
        return ("spike_down", min(0.9, abs(min_z) / 5))

    # v_recovery: V字回復
    if first_half < -15 and second_half > 15 and abs(price_change) < 20:
        confidence = min(0.9, (abs(first_half) + abs(second_half)) / 60)
        return ("v_recovery", confidence)

    # double_bottom: ダブルボトム
    if len(local_minima) >= 2 and abs(price_change) < 30:
        # 2つの底が類似した価格レベルにあるか確認
        if len(local_minima) >= 2:
            bottom1 = prices[local_minima[0]]
            bottom2 = prices[local_minima[1]]
            if abs(bottom1 - bottom2) / min(bottom1, bottom2) < 0.15:
                return ("double_bottom", 0.7)

    # uptrend: 上昇トレンド
    if price_change > 20 and r_squared > 0.5:
        confidence = min(0.95, r_squared * (1 + price_change / 100))
        return ("uptrend", confidence)

    # downtrend: 下降トレンド
    if price_change < -20 and r_squared > 0.5:
        confidence = min(0.95, r_squared * (1 + abs(price_change) / 100))
        return ("downtrend", confidence)

    # high_volatility: 高ボラティリティ
    if cv > 0.3:
        return ("high_volatility", min(0.9, cv))

    # low_volatility: 低ボラティリティ
    if cv < 0.1 and abs(price_change) < 15:
        return ("low_volatility", min(0.9, 1 - cv * 5))

    # range: レンジ相場
    if abs(price_change) < 10 and cv < 0.2:
        return ("range", min(0.8, 1 - abs(price_change) / 20))

    # gradual_change: 緩やかな変化
    if 10 <= abs(price_change) <= 20 and r_squared > 0.3:
        return ("gradual_change", min(0.7, r_squared))

    return None


def get_existing_files(output_dir: Path) -> dict[str, list[tuple[str, str, str]]]:
    """
    既存ファイルから期間情報を抽出。
    戻り値: {パターン名: [(ファイル名, base_token, quote_token), ...]}
    """
    existing = {p: [] for p in PATTERNS}

    for json_file in output_dir.glob("*.json"):
        match = re.match(r"([a-z_]+)-(\d+)\.json", json_file.name)
        if match:
            pattern = match.group(1)
            if pattern in existing:
                try:
                    with open(json_file, "r") as f:
                        # 先頭部分だけ読む
                        content = f.read(2000)
                        data = json.loads(content + "]}")  # 閉じて JSON としてパース
                        base_token = data.get("base_token", "")
                        quote_token = data.get("quote_token", "")
                        existing[pattern].append((json_file.name, base_token, quote_token))
                except (json.JSONDecodeError, KeyError):
                    existing[pattern].append((json_file.name, "", ""))

    return existing


def get_next_file_number(existing: dict[str, list], pattern: str) -> int:
    """次のファイル番号を取得"""
    max_num = 0
    for filename, _, _ in existing.get(pattern, []):
        match = re.match(rf"{pattern}-(\d+)\.json", filename)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def format_output_data(df: pd.DataFrame, pattern_match: PatternMatch) -> dict:
    """出力用 JSON データを生成"""
    decimals = pattern_match.decimals

    data_list = []
    for _, row in df.iterrows():
        rate = row["rate"]
        price = calculate_price(rate, decimals)
        data_list.append({
            "timestamp": row["timestamp"].isoformat(),
            "rate": f"{rate:.22f}",
            "price": f"{price:.49E}",
        })

    base_token, quote_token = pattern_match.token_pair.split("/")
    start_str = pattern_match.start_date.strftime("%Y-%m-%d")
    end_str = pattern_match.end_date.strftime("%Y-%m-%d")

    description = (
        f"{PATTERN_DESCRIPTIONS[pattern_match.pattern]}。"
        f"期間: {start_str} 〜 {end_str} ({WINDOW_DAYS}日間), "
        f"データ点数: {pattern_match.data_points}, "
        f"価格変化: {pattern_match.price_change:+.2f}%"
    )

    return {
        "description": description,
        "base_token": base_token,
        "quote_token": quote_token,
        "decimals": decimals,
        "data": data_list,
    }


def search_patterns(conn, target_patterns: list[str],
                    existing: dict[str, list], verbose: bool = False) -> list[PatternMatch]:
    """パターンを検索"""
    matches = []
    token_pairs = get_token_pairs_with_info(conn)

    if verbose:
        print(f"検索対象トークンペア: {len(token_pairs)} 組")

    for info in token_pairs:
        pair_name = f"{info.base_token}/{info.quote_token}"
        if verbose:
            print(f"\n検索中: {pair_name}")

        # スライディングウィンドウで検索
        current_start = info.min_timestamp
        window = timedelta(days=WINDOW_DAYS)
        step = timedelta(days=7)  # 1週間ずつずらす

        while current_start + window <= info.max_timestamp:
            current_end = current_start + window

            df = get_token_data(conn, info.base_token, info.quote_token,
                                current_start, current_end)

            if len(df) >= 100:
                result = analyze_pattern(df, info.decimals)
                if result:
                    pattern, confidence, metrics = result

                    if pattern in target_patterns and confidence > 0.5:
                        match = PatternMatch(
                            pattern=pattern,
                            token_pair=pair_name,
                            decimals=info.decimals,
                            start_date=current_start,
                            end_date=current_end,
                            data_points=len(df),
                            price_change=metrics["price_change"],
                            confidence=confidence,
                            metrics=metrics,
                        )
                        matches.append(match)

                        if verbose:
                            print(f"  発見: {pattern} ({current_start.date()} - {current_end.date()}), "
                                  f"信頼度: {confidence:.2f}, 価格変化: {metrics['price_change']:+.2f}%")

            current_start += step

    return matches


def save_pattern_data(conn, match: PatternMatch, output_dir: Path,
                      file_number: int, dry_run: bool = False) -> Optional[Path]:
    """パターンデータを保存"""
    base_token, quote_token = match.token_pair.split("/")
    df = get_token_data(conn, base_token, quote_token, match.start_date, match.end_date)

    if df.empty:
        return None

    output_data = format_output_data(df, match)
    filename = f"{match.pattern}-{file_number:02d}.json"
    filepath = output_dir / filename

    if dry_run:
        print(f"[dry-run] Would create: {filepath}")
        print(f"  Description: {output_data['description']}")
        return filepath

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL からパターンデータを取得"
    )
    parser.add_argument(
        "--pattern", "-p",
        choices=PATTERNS,
        help="特定のパターンのみを検索",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("tests/real"),
        help="出力先ディレクトリ (default: tests/real)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="検出のみ行い、ファイル出力しない",
    )
    parser.add_argument(
        "--max-per-pattern", "-m",
        type=int,
        default=5,
        help="パターンごとの最大ファイル数 (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細出力",
    )

    args = parser.parse_args()

    target_patterns = [args.pattern] if args.pattern else PATTERNS
    output_dir = args.output.resolve()

    if not output_dir.exists():
        if args.dry_run:
            print(f"[dry-run] Would create directory: {output_dir}")
        else:
            output_dir.mkdir(parents=True)

    print(f"パターン検索を開始: {', '.join(target_patterns)}")
    print(f"出力先: {output_dir}")

    try:
        conn = get_db_connection()
    except psycopg2.Error as e:
        print(f"DB 接続エラー: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        existing = get_existing_files(output_dir)
        print(f"\n既存ファイル数:")
        for p in target_patterns:
            count = len(existing[p])
            print(f"  {p}: {count}")

        matches = search_patterns(conn, target_patterns, existing, args.verbose)

        # 信頼度でソート
        matches.sort(key=lambda m: m.confidence, reverse=True)

        print(f"\n検出されたパターン: {len(matches)} 件")

        # パターンごとにグループ化して保存
        saved_count = {p: 0 for p in target_patterns}

        for match in matches:
            pattern = match.pattern
            current_count = len(existing[pattern]) + saved_count[pattern]

            if current_count >= args.max_per_pattern:
                continue

            file_number = get_next_file_number(existing, pattern) + saved_count[pattern]
            filepath = save_pattern_data(conn, match, output_dir, file_number, args.dry_run)

            if filepath:
                saved_count[pattern] += 1
                if not args.dry_run:
                    print(f"保存: {filepath.name}")

        print(f"\n保存完了:")
        for p in target_patterns:
            if saved_count[p] > 0:
                print(f"  {p}: {saved_count[p]} 件")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
