#!/usr/bin/env python3
"""
mlx_proxy.py - MLX LM Studio Dynamic Model Proxy
動的モデルロード・thinking制御・モデル一覧を提供するFastAPIプロキシ
"""

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ============================================================
# 設定
# ============================================================
MODELS_ROOT = Path("/path/to/your/LM_Studio_Models")  # ← 自分の環境に合わせて変更
MLX_SERVER_BIN = Path.home() / "mlx-server/bin/mlx_lm.server"
MLX_BACKEND_PORT = 18080   # mlx_lm.server が使うポート（内部）
PROXY_PORT = 8080          # このプロキシが使うポート
PROXY_HOST = "127.0.0.1"

# ============================================================
# 状態管理
# ============================================================
state: dict = {
    "process": None,        # subprocess.Popen
    "model_id": None,       # 現在ロード中のモデルID（パス）
    "enable_thinking": True,
    "loaded_at": None,
}

app = FastAPI(title="MLX Proxy", version="1.0.0")

# ============================================================
# モデル検索
# ============================================================
def find_all_models() -> list[dict]:
    """MODELS_ROOT 以下の全MLXモデルを返す"""
    models = []
    if not MODELS_ROOT.exists():
        return models
    for config in MODELS_ROOT.rglob("config.json"):
        model_path = config.parent
        model_id = str(model_path)
        # サブディレクトリ名から短縮名を生成
        rel = model_path.relative_to(MODELS_ROOT)
        short_name = str(rel)
        models.append({
            "id": model_id,
            "short_name": short_name,
            "path": model_id,
        })
    return sorted(models, key=lambda m: m["short_name"])


# ============================================================
# プロセス管理
# ============================================================
async def kill_backend():
    proc = state["process"]
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, proc.wait),
                timeout=10,
            )
        except asyncio.TimeoutError:
            proc.kill()
    state["process"] = None
    state["model_id"] = None
    state["loaded_at"] = None


async def wait_for_backend(timeout: float = 600.0):
    """バックエンドが起動するまで待つ"""
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"http://127.0.0.1:{MLX_BACKEND_PORT}/v1/models", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


async def start_backend(model_path: str, enable_thinking: bool):
    await kill_backend()

    chat_template_args = json.dumps({"enable_thinking": enable_thinking})
    cmd = [
        str(MLX_SERVER_BIN),
        "--model", model_path,
        "--port", str(MLX_BACKEND_PORT),
        "--host", "127.0.0.1",
        "--chat-template-args", chat_template_args,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state["process"] = proc
    state["model_id"] = model_path
    state["enable_thinking"] = enable_thinking
    state["loaded_at"] = time.time()

    ok = await wait_for_backend()
    if not ok:
        await kill_backend()
        raise RuntimeError(f"mlx_lm.server の起動がタイムアウトしました: {model_path}")


# ============================================================
# API: モデル管理
# ============================================================
class LoadRequest(BaseModel):
    model_id: str                          # パス or short_name
    enable_thinking: bool = False          # デフォルトはthinking off


@app.get("/v1/models")
async def list_models():
    """利用可能なモデル一覧（OpenAI互換）"""
    models = find_all_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m["short_name"],
                "object": "model",
                "path": m["path"],
                "loaded": m["path"] == state["model_id"],
            }
            for m in models
        ],
    }


@app.get("/v1/models/loaded")
async def loaded_model():
    """現在ロード中のモデル情報"""
    if not state["model_id"]:
        return {"loaded": False}
    return {
        "loaded": True,
        "model_id": state["model_id"],
        "enable_thinking": state["enable_thinking"],
        "loaded_at": state["loaded_at"],
    }


@app.post("/v1/models/load")
async def load_model(req: LoadRequest):
    """モデルをロード（ロード済みなら入れ替え）"""
    # short_name → フルパスに解決
    models = find_all_models()
    resolved = None
    for m in models:
        if req.model_id in (m["id"], m["short_name"], m["path"]):
            resolved = m["path"]
            break
    if not resolved:
        raise HTTPException(status_code=404, detail=f"モデルが見つかりません: {req.model_id}")

    try:
        await start_backend(resolved, req.enable_thinking)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "loaded",
        "model_id": resolved,
        "enable_thinking": req.enable_thinking,
    }


@app.post("/v1/models/unload")
async def unload_model():
    """モデルをアンロード"""
    if not state["model_id"]:
        return {"status": "already_unloaded"}
    await kill_backend()
    return {"status": "unloaded"}


# ============================================================
# API: チャット（バックエンドにプロキシ）
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not state["model_id"]:
        raise HTTPException(status_code=503, detail="モデルがロードされていません。/v1/models/load を先に呼んでください。")

    body = await request.body()

    # model フィールドをバックエンドのモデルIDに書き換え
    try:
        payload = json.loads(body)
        payload["model"] = state["model_id"]
        body = json.dumps(payload).encode()
    except Exception:
        pass

    backend_url = f"http://127.0.0.1:{MLX_BACKEND_PORT}/v1/chat/completions"
    stream = payload.get("stream", False)

    async with httpx.AsyncClient(timeout=300) as client:
        if stream:
            async def generate():
                async with client.stream("POST", backend_url,
                                         content=body,
                                         headers={"Content-Type": "application/json"}) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            r = await client.post(backend_url, content=body,
                                  headers={"Content-Type": "application/json"})
            return JSONResponse(content=r.json(), status_code=r.status_code)


# ============================================================
# ヘルスチェック
# ============================================================
@app.get("/health")
async def health():
    backend_ok = False
    if state["process"] and state["process"].poll() is None:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"http://127.0.0.1:{MLX_BACKEND_PORT}/v1/models")
                backend_ok = r.status_code == 200
        except Exception:
            pass
    return {
        "proxy": "ok",
        "backend": "ok" if backend_ok else "not_running",
        "model_id": state["model_id"],
        "enable_thinking": state["enable_thinking"],
    }


# ============================================================
# シャットダウン時にバックエンドも落とす
# ============================================================
@app.on_event("shutdown")
async def shutdown():
    await kill_backend()


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
