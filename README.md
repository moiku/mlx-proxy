# mlx-proxy

**Dynamic Model Management Proxy for MLX LM Studio Models**

`mlx_lm.server` only supports loading a single fixed model at startup. This proxy adds:

- **Dynamically load/unload models** via API calls
- **Toggle thinking on/off** per load with `enable_thinking`
- **Auto-discover all models** under your `LM_Studio_Models` directory
- **OpenAI-compatible API** — drop-in replacement for existing code

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14 Ventura or later
- Python 3.10+
- [mlx-lm](https://github.com/ml-explore/mlx-lm)

## Setup

### 1. Install mlx-lm

```bash
uv venv ~/mlx-server
source ~/mlx-server/bin/activate
uv pip install mlx-lm fastapi uvicorn httpx
```

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mlx-proxy
cd mlx-proxy
```

### 3. Edit configuration

Open `mlx_proxy.py` and update the settings at the top:

```python
MODELS_ROOT = Path("/path/to/your/LM_Studio_Models")  # ← Required: path to your models
MLX_SERVER_BIN = Path.home() / "mlx-server/bin/mlx_lm.server"  # ← Required: path to mlx_lm.server
MLX_BACKEND_PORT = 18080  # Internal port (no change needed)
PROXY_PORT = 8080         # Port exposed to clients
PROXY_HOST = "0.0.0.0"   # Use 0.0.0.0 for remote access, 127.0.0.1 for local only
```

> **MODELS_ROOT** and **MLX_SERVER_BIN** must be set to match your environment.
>
> Examples:
> - LM Studio default: `~/.lmstudio/models`
> - External SSD: `/Volumes/MySSD/LM_Studio_Models`

### 4. Start the proxy

```bash
source ~/mlx-server/bin/activate
python mlx_proxy.py
```

### 5. Auto-start with launchd (optional)

```bash
# Edit the plist to match your username and paths
cp com.yourname.mlx-proxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.yourname.mlx-proxy.plist
```

## Usage

### List available models

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
      "path": "/path/to/LM_Studio_Models/mlx-community/Qwen3.5-9B-MLX-4bit",
      "loaded": false
    }
  ]
}
```

### Load a model

```bash
# Load with thinking disabled (recommended default)
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/Qwen3.5-9B-MLX-4bit", "enable_thinking": false}'

# Load with thinking enabled
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/Qwen3.5-9B-MLX-4bit", "enable_thinking": true}'
```

### Chat (OpenAI-compatible)

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

> The `model` field can be anything — it is automatically replaced with the loaded model.

### Using with the OpenAI Python SDK

```python
from openai import OpenAI
import requests

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

# Load a model first
requests.post("http://localhost:8080/v1/models/load", json={
    "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "enable_thinking": False
})

# Chat (non-streaming)
response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming directly from the backend

Due to a known limitation (see Notes), streaming via the proxy returns a premature connection close. For streaming, connect directly to the backend port after loading via the proxy:

```python
import json
import requests

# Step 1: load via proxy (thinking control requires the proxy)
requests.post("http://localhost:8080/v1/models/load", json={
    "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "enable_thinking": False
})

# Step 2: get the full model path from the proxy
health = requests.get("http://localhost:8080/health").json()
model_path = health["model_id"]  # e.g. "/Volumes/.../Qwen3.5-9B-MLX-4bit"

# Step 3: stream directly from the backend (mlx_lm.server, HTTP/1.0)
url = "http://127.0.0.1:18080/v1/chat/completions"
payload = {
    "model": model_path,
    "messages": [{"role": "user", "content": "Hello!"}],
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

### Unload the model

```bash
curl -X POST http://localhost:8080/v1/models/unload
```

### Health check

```bash
curl http://localhost:8080/health
```

```json
{
  "proxy": "ok",
  "backend": "ok",
  "model_id": "/path/to/Qwen3.5-9B-MLX-4bit",
  "enable_thinking": false
}
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List all available models |
| GET | `/v1/models/loaded` | Get currently loaded model info |
| POST | `/v1/models/load` | Load a model |
| POST | `/v1/models/unload` | Unload the current model |
| POST | `/v1/chat/completions` | Chat (OpenAI-compatible) |
| GET | `/health` | Health check |

## Architecture

```
Client / External App
        ↓ :8080 (PROXY_PORT)
  mlx_proxy.py (FastAPI)
        ↓ :18080 (MLX_BACKEND_PORT)
  mlx_lm.server (MLX backend)
        ↓
  /path/to/LM_Studio_Models/
```

Model switching is achieved by restarting the backend process with the new model and `chat-template-args`.

## Notes

- Model switching restarts the backend — allow a few seconds for loading
- Only one model can be loaded at a time
- If using an external SSD, the drive may not be mounted in time on boot. Re-trigger the proxy with: `launchctl kickstart gui/$(id -u)/com.yourname.mlx-proxy`

### Streaming limitation

`mlx_lm.server` responds with **HTTP/1.0** (no chunked transfer encoding — it signals end-of-stream by closing the connection). When the proxy forwards this as a chunked HTTP/1.1 stream to the client, libraries such as `requests` raise `ChunkedEncodingError: Response ended prematurely`.

**Workaround options:**

| Use case | Recommendation |
|----------|---------------|
| Non-streaming (simple) | Use the proxy endpoint normally — works fine |
| Streaming | Load via proxy, then stream directly from `http://127.0.0.1:18080` (see example above) |

### Impact of `enable_thinking`

For Thinking Models (e.g. Qwen3.5), `enable_thinking` has a dramatic effect on latency:

| Setting | Typical total response time (9B model) |
|---------|----------------------------------------|
| `enable_thinking: true` | 2–5 minutes (reasoning phase dominates) |
| `enable_thinking: false` | < 1 second |

Use `enable_thinking: false` (the default) for interactive use. Enable thinking only when deep reasoning is needed and latency is not a concern.

## License

MIT
