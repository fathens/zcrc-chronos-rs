# Contributing

## テストコードの分離

### ルール

以下の **両方** を満たすファイルは、テストコードを別ファイルに分離する。

1. テストコード（`#[cfg(test)] mod tests { ... }` ブロック）がファイル全体の **1/4 超**
2. テストコードが **100 行超**

### 分離方法

`foo.rs` を `foo.rs` + `foo/tests.rs` に分割する。`mod.rs` は使わない。

**変更前:**

```
src/
  foo.rs          # プロダクションコード + テスト
```

**変更後:**

```
src/
  foo.rs          # プロダクションコード + #[cfg(test)] mod tests;
  foo/
    tests.rs      # テストモジュールの中身（mod tests { } の内側だけ）
```

**`foo.rs` の末尾:**

```rust
#[cfg(test)]
mod tests;
```

**`foo/tests.rs`:**

```rust
use super::*;

#[test]
fn test_example() {
    // ...
}
```
