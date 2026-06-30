# mlx-proxy 変更ログ

---

## 2026-03-30 — VLM モード追加・Tailscale 公開対応

### 背景

`validate_nm_vision.py`（nm シリーズ画像あり問題の品質検証）で、
Qwen3.5-27B を Vision モードで使う際に `enable_thinking` を無効化できない問題があった。

- mlx_vlm を直接呼ぶ場合、`/no_think` をプロンプトに入れても効かない
- thinking の ON/OFF はモデルロード時の `apply_chat_template(..., enable_thinking=False)` で制御する必要がある
- mlx-proxy のように「ロード時に設定を固定する」仕組みが VLM にも必要

### 変更内容

#### `mlx_proxy.py`

**1. VLM モードの追加**

`/v1/models/load` に `mode` フィールドを追加。

```json
{
  "model_id": "mlx-community/Qwen3.5-27B-bf16",
  "enable_thinking": false,
  "mode": "vlm"
}
```

| `mode` | 動作 | 用途 |
|--------|------|------|
| `"llm"`（デフォルト） | mlx_lm.server をサブプロセス起動 | テキストのみ |
| `"vlm"` | mlx_vlm をインプロセスでロード | 画像 + テキスト |

**2. `enable_thinking` の VLM 対応**

VLM モードでは `apply_chat_template(..., enable_thinking=False)` をロード時の設定として保持し、
推論のたびに適用する。これにより thinking ON/OFF がプロンプトレベルで制御される。

**3. 画像入力の対応（OpenAI 互換形式）**

`/v1/chat/completions` で以下の形式をサポート:

```json
{
  "type": "image_url",
  "image_url": {"url": "file:///absolute/path/to/image.png"}
}
```

```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/png;base64,<base64文字列>"}
}
```

複数画像も対応（リストの順に渡される）。

**4. `/health`・`/v1/models/loaded` に `mode` フィールドを追加**

```json
{
  "proxy": "ok",
  "backend": "ok",
  "model_id": "...",
  "enable_thinking": false,
  "mode": "vlm"
}
```

**5. インポート追加**

`base64`, `tempfile`, `uuid` を標準ライブラリからインポート追加。

#### 追加ファイル

- `rs-math-textbook/mlx_proxy_manual.md` — 学生向け利用マニュアル（Tailscale 経由）

#### 関連スクリプトの変更

- `rs-math-textbook/validate_nm_vision.py`
  - mlx_vlm 直接呼び出し → mlx-proxy API 経由に変更
  - ロード時: `mode=vlm, enable_thinking=False` を指定
  - 画像: `file://` URL 形式で送信

### アーキテクチャ

```
クライアント（学生PC / スクリプト）
    │  HTTP (Tailscale: kdrive.tail4a4e5b.ts.net:8080)
    ▼
[mlx-proxy :8080]  ← launchd 常駐 (com.gendo.mlx-proxy)
    │
    ├─ mode=llm → [mlx_lm.server :18080] サブプロセス
    │
    └─ mode=vlm → mlx_vlm インプロセス推論
```

### 技術メモ

- mlx_vlm の `apply_chat_template` は `**kwargs` 経由でトークナイザの `apply_chat_template` に
  `enable_thinking` を渡す。これが Qwen3.5 チャットテンプレートの thinking 制御に効く。
- VLM 推論は `run_in_executor(None, ...)` でスレッドプールで実行（FastAPI のイベントループをブロックしない）。
- base64 画像は一時ファイルに展開 → 推論後に削除。
- LLM モードのストリーミングは従来通り動作（VLM モードは非ストリーミングのみ）。

---

## 2026-03-28 以前（初期構築）

- `Initial commit`: FastAPI プロキシ基盤、`/v1/models/load`・`/v1/chat/completions`・`/health` 実装
- ロードタイムアウトを 1 分 → 10 分に延長
- launchd plist (`com.gendo.mlx-proxy`) で常駐化、`KeepAlive=true` で自動再起動
- `--chat-template-args '{"enable_thinking": false}'` で mlx_lm.server の thinking 制御を実現
