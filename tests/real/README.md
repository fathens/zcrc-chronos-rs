# Real Data Test Files

予測アルゴリズムの精度検証に使用する実際の価格データ。

## データ概要

- **ソース**: NEAR Protocol 上のトークン価格データ
- **期間**: 各ファイル約31日間
- **データ点数**: 各ファイル数百〜数千点

## データ構造

各 JSON ファイルは以下の構造を持つ:

```json
{
  "description": "パターンの説明（日本語）",
  "base_token": "blackdragon.tkn.near",
  "quote_token": "wrap.near",
  "decimals": 24,
  "data": [
    {
      "timestamp": "2025-08-21T08:09:43.671895",
      "rate": "249613395950990997870559630494400.0000000000000000000000",
      "price": "4.0061952452116777871309959660363577899314011550652E-9"
    }
  ]
}
```

| フィールド | 説明 |
|-----------|------|
| `description` | パターンの説明、期間、価格変化率 |
| `base_token` | ベーストークンのアドレス |
| `quote_token` | クォートトークンのアドレス |
| `decimals` | トークンの小数桁数 |
| `data[].timestamp` | データポイントのタイムスタンプ (ISO 8601) |
| `data[].rate` | 生のレート値 |
| `data[].price` | 計算された価格 (base/quote) |

## ファイル命名規則

```
{pattern}-{番号}.json
```

例: `uptrend-01.json`, `uptrend-02.json`, ...

## パターン一覧

| パターン | 説明 |
|---------|------|
| `uptrend` | 上昇トレンド |
| `downtrend` | 下降トレンド |
| `low_volatility` | 低ボラティリティ |
| `high_volatility` | 高ボラティリティ |
| `range` | レンジ相場 |
| `gradual_change` | 緩やかな変化 |
| `double_bottom` | ダブルボトム |
| `spike_up` | 急上昇 |
| `spike_down` | 急落 |
| `v_recovery` | V字回復 |

## データベースからの取得

データは PostgreSQL から取得できる。

### 接続情報

| 項目 | 値 |
|-----|-----|
| Host | `localhost` |
| Port | `5432` |
| Database | `postgres` |
| User | `readonly` |
| Password | `readonly` |

### テーブル構造

テーブル: `public.token_rates`

| カラム | 型 | 説明 |
|-------|-----|------|
| `id` | integer | 主キー (auto increment) |
| `base_token` | varchar | ベーストークンのアドレス |
| `quote_token` | varchar | クォートトークンのアドレス |
| `rate` | numeric | レート値 |
| `timestamp` | timestamp | データポイントのタイムスタンプ |
| `decimals` | smallint | トークンの小数桁数 |

### 価格の計算

JSON ファイル内の `price` は以下の式で計算される:

```
price = 10^decimals / rate
```

### 接続例

```bash
psql -h localhost -p 5432 -U readonly -d postgres
```

```rust
// Rust (sqlx)
let pool = PgPoolOptions::new()
    .connect("postgres://readonly:readonly@localhost:5432/postgres")
    .await?;
```

### クエリ例

```sql
-- 特定トークンペアのデータを取得
SELECT timestamp, rate, decimals
FROM token_rates
WHERE base_token = 'blackdragon.tkn.near'
  AND quote_token = 'wrap.near'
ORDER BY timestamp;

-- 期間を指定して取得
SELECT timestamp, rate, decimals
FROM token_rates
WHERE base_token = 'blackdragon.tkn.near'
  AND quote_token = 'wrap.near'
  AND timestamp BETWEEN '2025-08-01' AND '2025-09-01'
ORDER BY timestamp;
```
