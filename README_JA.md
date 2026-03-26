# mlx-proxy

**MLX LM Studio Models用 動的モデル管理プロキシ**

`mlx_lm.server` はモデルを固定して起動する仕様ですが、このプロキシを使うと：

- 複数モデルを **APIコールで動的にロード/アンロード**
- `enable_thinking` で **thinking on/off を切り替え**
- `LM_Studio_Models` 配下の **全モデルを自動検出**
- OpenAI互換APIとして動作するので **既存のコードをそのまま利用可能**

## 動作環境

- Apple Silicon Mac（M1/M2/M3/M4）
- macOS 14 Ventura以降
- Python 3.10+
- [mlx-lm](https://github.com/ml-explore/mlx-lm)

## セットアップ

### 1. mlx-lmのインストール

```bash
uv venv ~/mlx-server
source ~/mlx-server/bin/activate
uv pip install mlx-lm fastapi uvicorn httpx
```

### 2. mlx-proxyの配置

```bash
git clone https://github.com/YOUR_USERNAME/mlx-proxy
cd mlx-proxy
```

### 3. 設定の編集

`mlx_proxy.py` 冒頭の設定を環境に合わせて変更：

```python
MODELS_ROOT = Path("/path/to/your/LM_Studio_Models")  # ← 必須: 自分の環境に合わせて変更
MLX_SERVER_BIN = Path.home() / "mlx-server/bin/mlx_lm.server"  # ← 必須: mlx_lm.serverのパスに変更
MLX_BACKEND_PORT = 18080  # 内部ポート（変更不要）
PROXY_PORT = 8080         # 外部公開ポート
PROXY_HOST = "0.0.0.0"   # 外部から叩く場合は0.0.0.0、ローカルのみは127.0.0.1
```

> **MODELS_ROOT** と **MLX_SERVER_BIN** は必ず自分の環境に合わせて変更してください。
>
> 例：
> - LM Studioのデフォルト保存先: `~/.lmstudio/models`
> - 外付けSSDに保存している場合: `/Volumes/MySSD/LM_Studio_Models`

### 4. 起動

```bash
source ~/mlx-server/bin/activate
python mlx_proxy.py
```

### 5. 自動起動（launchd）

```bash
# plistを編集してユーザー名・パスを自分の環境に合わせる
cp com.yourname.mlx-proxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.yourname.mlx-proxy.plist
```

## 使い方

### モデル一覧

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen3.5-9B-MLX-4bit",
      "object": "model",
      "path": "/Volumes/MySSD/LM_Studio_Models/mlx-community/Qwen3.5-9B-MLX-4bit",
      "loaded": false
    },
    ...
  ]
}
```

### モデルのロード

```bash
# thinking off（デフォルト推奨）
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/Qwen3.5-9B-MLX-4bit", "enable_thinking": false}'

# thinking on
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/Qwen3.5-9B-MLX-4bit", "enable_thinking": true}'
```

### チャット（OpenAI互換）

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "こんにちは"}]
  }'
```

> `model` フィールドは何でもOKです（ロード済みモデルに自動で書き換わります）

### Pythonから使う

```python
from openai import OpenAI
import requests

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

# モデルのロード
requests.post("http://localhost:8080/v1/models/load", json={
    "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "enable_thinking": False
})

# チャット（非ストリーミング）
response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "こんにちは"}]
)
print(response.choices[0].message.content)
```

### ストリーミングする場合はバックエンドに直接接続

後述の制限により、プロキシ経由のストリーミングは接続が途中で切れます。
ストリーミングが必要な場合は、プロキシでモデルをロードしてからバックエンドに直接接続してください：

```python
import json
import requests

# Step 1: プロキシ経由でロード（thinking制御にはプロキシが必要）
requests.post("http://localhost:8080/v1/models/load", json={
    "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "enable_thinking": False
})

# Step 2: プロキシのhealthからフルパスを取得
health = requests.get("http://localhost:8080/health").json()
model_path = health["model_id"]  # 例: "/Volumes/.../Qwen3.5-9B-MLX-4bit"

# Step 3: バックエンド (mlx_lm.server / HTTP/1.0) に直接ストリーミング
url = "http://127.0.0.1:18080/v1/chat/completions"
payload = {
    "model": model_path,
    "messages": [{"role": "user", "content": "こんにちは"}],
    "stream": True,
}
with requests.post(url, json=payload, stream=True) as resp:
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode() if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:]
        if data_str.strip() == "[DONE]":
            break
        chunk = json.loads(data_str)
        content = chunk["choices"][0]["delta"].get("content", "")
        if content:
            print(content, end="", flush=True)
```

### モデルのアンロード

```bash
curl -X POST http://localhost:8080/v1/models/unload
```

### ヘルスチェック

```bash
curl http://localhost:8080/health
```

```json
{
  "proxy": "ok",
  "backend": "ok",
  "model_id": "/Volumes/.../Qwen3.5-9B-MLX-4bit",
  "enable_thinking": false
}
```

## エンドポイント一覧

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/v1/models` | 利用可能なモデル一覧 |
| GET | `/v1/models/loaded` | 現在ロード中のモデル情報 |
| POST | `/v1/models/load` | モデルのロード |
| POST | `/v1/models/unload` | モデルのアンロード |
| POST | `/v1/chat/completions` | チャット（OpenAI互換） |
| GET | `/health` | ヘルスチェック |

## 仕組み

```
クライアント / 外部アプリ
        ↓ :8080 (PROXY_PORT)
  mlx_proxy.py (FastAPI)
        ↓ :18080 (MLX_BACKEND_PORT)
  mlx_lm.server (MLXバックエンド)
        ↓
  /Volumes/MySSD/LM_Studio_Models/
```

モデルのロード時にバックエンドプロセスを再起動することでモデル切り替えを実現しています。

## 注意事項

- モデルの切り替え時はバックエンドが再起動するため、ロード完了まで数十秒かかります
- 同時に使えるモデルは1つだけです
- 外付けSSDを使っている場合、Mac起動直後にSSDのマウントが間に合わないことがあります。その場合は `launchctl kickstart gui/$(id -u)/com.yourname.mlx-proxy` で再起動してください

### ストリーミングの制限

`mlx_lm.server` は **HTTP/1.0** でストリーミング応答します（chunked transfer encoding なし、接続クローズで終端）。
FastAPI + httpx でプロキシすると、クライアント側で `ChunkedEncodingError: Response ended prematurely` が発生します。

**回避策：**

| ユースケース | 推奨 |
|------------|------|
| 非ストリーミング | プロキシ経由で普通に使えばOK |
| ストリーミング | プロキシでロード後、`http://127.0.0.1:18080` に直接接続（上記のサンプルを参照） |

### `enable_thinking` の効果

Thinking Model（Qwen3.5など）では `enable_thinking` の設定が応答速度に大きく影響します：

| 設定 | 典型的な総応答時間（9Bモデル） |
|------|------------------------------|
| `enable_thinking: true` | 2〜5分（reasoning フェーズが支配的） |
| `enable_thinking: false` | 1秒以下 |

インタラクティブな用途では `enable_thinking: false`（デフォルト）を推奨します。
深い推論が必要で遅延を許容できる場合のみ `true` にしてください。

## ライセンス

MIT
