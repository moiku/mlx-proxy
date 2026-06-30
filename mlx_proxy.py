#!/usr/bin/env python3
"""
mlx_proxy.py - MLX LM Studio Dynamic Model Proxy  v1.1
動的モデルロード・thinking制御・モデル一覧・直接スコアリング・MCPエージェントループを提供する FastAPI プロキシ

mode:
  "llm"   (default) - mlx_lm.server をサブプロセスで起動（テキストのみ）
  "vlm"             - mlx_vlm をインプロセスでロード（画像+テキスト対応）
  "score"           - mlx_lm をインプロセスでロードし次トークンの logprob を直接返す

新エンドポイント (v1.1):
  POST /v1/score          - 指定トークンの logprob を直接返す（top-N 取りこぼし解決）
  POST /v1/agent/chat     - MCP ツール呼び出しを含むエージェントループ
  GET  /v1/agent/tools    - 登録済み MCP ツール一覧
"""

import asyncio
import base64
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ============================================================
# 設定
# ============================================================
MODELS_ROOT = Path("/Volumes/SanDiskSSD/LM_Studio_Models")
MLX_BACKEND_PORT = 18080   # mlx_lm.server が使うポート（内部）
PROXY_PORT = 8080          # このプロキシが使うポート
PROXY_HOST = "0.0.0.0"

# MCP クライアント設定ファイル（存在すれば /v1/agent/chat で使用）
MCP_CLIENTS_CONFIG = Path(__file__).parent / "mcp_clients.json"

# エージェントループの最大反復回数
AGENT_MAX_ITERATIONS = 8

# ============================================================
# 状態管理
# ============================================================
state: dict = {
    # LLM モード
    "process": None,          # subprocess.Popen (LLMモード)
    # VLM モード
    "vlm_model": None,        # mlx_vlm model
    "vlm_processor": None,    # mlx_vlm processor
    "vlm_config": None,       # mlx_vlm config
    "vlm_draft_model": None,  # MTP drafter (mlx_vlm speculative decoding)
    "vlm_draft_kind": None,   # drafter kind ("mtp", "dflash", etc.)
    # Score モード
    "score_model": None,      # mlx_lm model
    "score_tokenizer": None,  # mlx_lm tokenizer (TokenizerWrapper)
    # 共通
    "model_id": None,         # 現在ロード中のモデルID（パス）
    "enable_thinking": False,
    "mode": "llm",            # "llm" | "vlm" | "score"
    "loaded_at": None,
}

# MCP セッション管理（/v1/agent/chat 用）
_mcp_sessions: dict = {}   # tool_name -> (session, srv_name, tool_name)
_mcp_tools: list = []      # OpenAI tools 形式のスキーマリスト

app = FastAPI(title="MLX Proxy", version="1.1.0")

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
    """全モードのバックエンドを停止しメモリを解放する"""
    # LLM モード: サブプロセスを終了
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

    # VLM モード: モデル参照を解放
    state["vlm_model"] = None
    state["vlm_processor"] = None
    state["vlm_config"] = None
    state["vlm_draft_model"] = None
    state["vlm_draft_kind"] = None

    # Score モード: モデル参照を解放
    state["score_model"] = None
    state["score_tokenizer"] = None

    state["model_id"] = None
    state["mode"] = "llm"
    state["loaded_at"] = None

    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    await asyncio.sleep(3.0)


async def wait_for_backend(timeout: float = 600.0):
    """LLMモードのバックエンドが起動するまで待つ"""
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
    """LLM モード: mlx_lm server をサブプロセスで起動

    sys.executable を使い、現在の venv の mlx_lm を確実に使う（パスドリフト防止）。
    MLX_SERVER_CMD 環境変数でコマンドをオーバーライド可能。
    """
    await kill_backend()

    chat_template_args = json.dumps({"enable_thinking": enable_thinking})
    mlx_server_cmd = os.environ.get("MLX_SERVER_CMD", "")
    if mlx_server_cmd:
        cmd = [mlx_server_cmd]
    else:
        # sys.executable -m mlx_lm server で現在の venv を使う
        cmd = [sys.executable, "-m", "mlx_lm", "server"]

    cmd += [
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
    state["mode"] = "llm"
    state["loaded_at"] = time.time()

    ok = await wait_for_backend()
    if not ok:
        await kill_backend()
        raise RuntimeError(f"mlx_lm.server の起動がタイムアウトしました: {model_path}")


async def start_vlm_backend(model_path: str, enable_thinking: bool,
                            draft_model_id: str = None, draft_kind: str = None):
    """VLM モード: mlx_vlm でモデルをインプロセスにロード。
    draft_model_id を指定すると MTP/dflash ドラフターも同時にロードする。
    """
    await kill_backend()

    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config as vlm_load_config

    loop = asyncio.get_event_loop()
    model, processor = await loop.run_in_executor(None, vlm_load, model_path)
    config = await loop.run_in_executor(None, vlm_load_config, model_path)

    draft_model_obj = None
    resolved_kind = draft_kind
    if draft_model_id:
        # ドラフターパスを解決（short_name or フルパス）
        models = find_all_models()
        draft_path = draft_model_id
        for m in models:
            if draft_model_id in (m["id"], m["short_name"], m["path"]):
                draft_path = m["path"]
                break

        from mlx_vlm.speculative.drafters import load_drafter, DRAFTER_KIND_BY_MODEL_TYPE
        draft_model_obj, resolved_kind = await loop.run_in_executor(
            None, lambda: load_drafter(draft_path, kind=draft_kind)
        )

    state["vlm_model"] = model
    state["vlm_processor"] = processor
    state["vlm_config"] = config
    state["vlm_draft_model"] = draft_model_obj
    state["vlm_draft_kind"] = resolved_kind
    state["model_id"] = model_path
    state["enable_thinking"] = enable_thinking
    state["mode"] = "vlm"
    state["loaded_at"] = time.time()


async def start_score_backend(model_path: str):
    """Score モード: mlx_lm でモデルをインプロセスにロード（logits 直接アクセス用）"""
    await kill_backend()

    from mlx_lm import load as lm_load

    loop = asyncio.get_event_loop()
    model, tokenizer = await loop.run_in_executor(None, lm_load, model_path)

    state["score_model"] = model
    state["score_tokenizer"] = tokenizer
    state["model_id"] = model_path
    state["enable_thinking"] = False
    state["mode"] = "score"
    state["loaded_at"] = time.time()


# ============================================================
# API: モデル管理
# ============================================================
class LoadRequest(BaseModel):
    model_id: str                          # パス or short_name
    enable_thinking: bool = False          # デフォルトはthinking off
    mode: str = "llm"                      # "llm" | "vlm" | "score"
    draft_model_id: Optional[str] = None  # VLMモード専用: MTPドラフターのパス or short_name
    draft_kind: Optional[str] = None      # VLMモード専用: "mtp" | "dflash" | "eagle3" (省略時は自動判定)


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
        "mode": state["mode"],
        "loaded_at": state["loaded_at"],
    }


@app.post("/v1/models/load")
async def load_model(req: LoadRequest):
    """モデルをロード（ロード済みなら入れ替え）"""
    if req.mode not in ("llm", "vlm", "score"):
        raise HTTPException(status_code=400, detail=f"mode は 'llm', 'vlm', 'score' のいずれかを指定: {req.mode}")

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
        if req.mode == "vlm":
            await start_vlm_backend(resolved, req.enable_thinking,
                                    draft_model_id=req.draft_model_id,
                                    draft_kind=req.draft_kind)
        elif req.mode == "score":
            await start_score_backend(resolved)
        else:
            await start_backend(resolved, req.enable_thinking)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "loaded",
        "model_id": resolved,
        "enable_thinking": req.enable_thinking,
        "mode": req.mode,
    }


@app.post("/v1/models/unload")
async def unload_model():
    """モデルをアンロード"""
    if not state["model_id"]:
        return {"status": "already_unloaded"}
    await kill_backend()
    return {"status": "unloaded"}


# ============================================================
# VLM 推論ヘルパー
# ============================================================
def _extract_images_from_messages(messages: list) -> tuple[list[str], list[dict], list[str]]:
    """
    メッセージから画像パスを抽出し、テキストのみのメッセージリストを返す。
    対応形式:
      - {"type": "image_url", "image_url": {"url": "file:///path/to/img.png"}}
      - {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    images = []
    clean_messages = []
    tmp_files = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type", "")
                if t == "text":
                    text_parts.append(item.get("text", ""))
                elif t == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    if url.startswith("file://"):
                        images.append(url[7:])
                    elif url.startswith("data:image"):
                        header, data = url.split(",", 1)
                        ext = header.split("/")[1].split(";")[0]
                        tmp = tempfile.mktemp(suffix=f".{ext}")
                        with open(tmp, "wb") as f:
                            f.write(base64.b64decode(data))
                        images.append(tmp)
                        tmp_files.append(tmp)
            clean_messages.append({"role": msg.get("role", "user"),
                                    "content": "\n".join(text_parts)})
        else:
            clean_messages.append(msg)

    return images, clean_messages, tmp_files


def _vlm_generate_sync(body: dict) -> dict:
    """VLMモードでの推論（同期・ブロッキング）"""
    import mlx.core as mx
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 1024)
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)

    # MLX の乱数状態はスレッドローカルかつ生成ストリームに束縛されるため、
    # run_in_executor スレッドでは mx.random.seed が生成サンプリングに反映されない。
    # seed を generate_step まで明示的に渡すと seed 由来の明示的 key でサンプリングされ、
    # スレッド非依存になる。body で seed 指定があれば再現用に尊重し、なければ乱数化する。
    seed = body.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")

    images, clean_messages, tmp_files = _extract_images_from_messages(messages)

    model = state["vlm_model"]
    processor = state["vlm_processor"]
    config = state["vlm_config"]
    enable_thinking = state["enable_thinking"]
    draft_model = state["vlm_draft_model"]
    draft_kind = state["vlm_draft_kind"]

    num_images = len(images)
    prompt = apply_chat_template(
        processor, config, clean_messages,
        num_images=num_images,
        enable_thinking=enable_thinking,
    )

    img_arg = images[0] if len(images) == 1 else (images if images else None)

    generate_kwargs = dict(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        verbose=False,
    )
    if draft_model is not None:
        generate_kwargs["draft_model"] = draft_model
        generate_kwargs["draft_kind"] = draft_kind

    t0 = time.time()
    out = vlm_generate(
        model, processor, prompt,
        image=img_arg,
        **generate_kwargs,
    )
    elapsed = time.time() - t0

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    text = out.text if hasattr(out, "text") else str(out)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": state["model_id"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_sec": round(elapsed, 2),
            "temperature_used": temperature,
            "seed_used": seed,
        },
    }


# ============================================================
# Score モード: 指定トークンの logprob を直接返す
# ============================================================
class ScoreRequest(BaseModel):
    # プロンプト指定方法（どちらか一方）
    prompt: Optional[str] = None           # 生文字列プロンプト
    messages: Optional[list] = None        # chat messages（apply_chat_template=True 時に使用）
    # 評価対象の候補文字列リスト
    candidates: list[str]
    # apply_chat_template=True の場合 messages を chat template に通す
    apply_chat_template: bool = False
    # multi_token 候補の処理方法
    # "first": 最初のトークンの logprob のみ（高速・単一トークン候補では完全に正確）
    # "sum"  : teacher-forcing で全トークンの logprob を合算（遅いが複数トークン候補に正確）
    multi_token: str = "first"


def _score_sync(req_dict: dict) -> dict:
    """Score モードでの logprob 計算（同期・ブロッキング）"""
    import mlx.core as mx

    model = state["score_model"]
    tokenizer = state["score_tokenizer"]
    if model is None or tokenizer is None:
        raise RuntimeError("score モデルがロードされていません")

    # プロンプトのトークン化
    apply_tmpl = req_dict.get("apply_chat_template", False)
    if apply_tmpl:
        messages = req_dict.get("messages") or []
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        else:
            # フォールバック: user メッセージを結合
            text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
            prompt_ids = tokenizer.encode(text)
    else:
        text = req_dict.get("prompt") or ""
        prompt_ids = tokenizer.encode(text)

    # 1回 forward pass で次トークンの logits を取得（generate.py 400-420 と同じパターン）
    t0 = time.time()
    x = mx.array(prompt_ids)[None]        # (1, T)
    logits = model(x)                     # (1, T, V)
    logits = logits[:, -1, :]             # (1, V) 最終位置の next-token logits
    logprobs_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # log_softmax
    logprobs_all = logprobs_all[0]        # (V,)
    mx.eval(logprobs_all)                 # MLX lazy eval を確定させる
    elapsed = time.time() - t0

    candidates = req_dict.get("candidates", [])
    multi_token = req_dict.get("multi_token", "first")

    results = []
    for cand in candidates:
        try:
            token_ids = tokenizer.encode(cand, add_special_tokens=False)
        except Exception as e:
            results.append({
                "text": cand,
                "token_ids": [],
                "logprob": None,
                "prob_normalized": None,
                "error": f"encode failed: {e}",
            })
            continue

        if len(token_ids) == 0:
            results.append({
                "text": cand,
                "token_ids": [],
                "logprob": None,
                "prob_normalized": None,
                "error": "empty token sequence",
            })
            continue

        if len(token_ids) == 1 or multi_token == "first":
            # 単一トークン or "first" モード: 最初のトークンの logprob
            lp = float(logprobs_all[token_ids[0]].item())
        else:
            # "sum" モード: teacher-forcing で全トークンの logprob を合算
            lp = float(logprobs_all[token_ids[0]].item())
            ids_so_far = list(prompt_ids) + [token_ids[0]]
            for tid in token_ids[1:]:
                xf = mx.array(ids_so_far)[None]
                lf = model(xf)
                lf = lf[:, -1, :]
                lp_f = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
                mx.eval(lp_f)
                lp += float(lp_f[0, tid].item())
                ids_so_far.append(tid)

        results.append({
            "text": cand,
            "token_ids": token_ids,
            "logprob": lp,
            "prob_normalized": None,  # 後で計算
        })

    # 候補集合内 softmax (prob_normalized) を計算
    valid_lps = [r["logprob"] for r in results if r["logprob"] is not None]
    if valid_lps:
        max_lp = max(valid_lps)
        sum_exp = sum(math.exp(lp - max_lp) for lp in valid_lps)
        log_sum_exp = max_lp + math.log(sum_exp)
        for r in results:
            if r["logprob"] is not None:
                r["prob_normalized"] = math.exp(r["logprob"] - log_sum_exp)

    # argmax（prob_normalized 最大の候補）
    valid = [r for r in results if r["prob_normalized"] is not None]
    argmax = max(valid, key=lambda r: r["prob_normalized"])["text"] if valid else None

    return {
        "object": "score",
        "model_id": state["model_id"],
        "prompt_tokens": len(prompt_ids),
        "candidates": results,
        "argmax": argmax,
        "elapsed_sec": round(elapsed, 2),
    }


@app.post("/v1/score")
async def score(request: Request):
    """指定トークンの logprob を直接返す（top-N 取りこぼし問題を根絶）

    事前に POST /v1/models/load {"mode": "score"} でモデルをロードしてください。
    """
    if state["mode"] != "score":
        raise HTTPException(
            status_code=409,
            detail=(
                f"score モードのモデルをロードしてください: "
                f"POST /v1/models/load {{\"mode\":\"score\",...}}"
                f"（現在 mode={state['mode']!r}）"
            )
        )

    payload = await request.json()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _score_sync, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"スコアリングエラー: {e}")
    return JSONResponse(content=result)


# ============================================================
# API: チャット（バックエンドにプロキシ）
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not state["model_id"]:
        raise HTTPException(status_code=503, detail="モデルがロードされていません。/v1/models/load を先に呼んでください。")
    if state["mode"] == "score":
        raise HTTPException(status_code=409, detail="score モードでは /v1/chat/completions は使えません。llm または vlm モードでロードしてください。")

    payload = await request.json()

    # VLM モード: インプロセスで推論
    if state["mode"] == "vlm":
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, _vlm_generate_sync, payload)
        except Exception as e:
            await kill_backend()
            raise HTTPException(status_code=500, detail=f"VLM推論エラー: {e}")
        return JSONResponse(content=result)

    # LLM モード: mlx_lm.server バックエンドにプロキシ
    payload["model"] = state["model_id"]
    body = json.dumps(payload).encode()

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
# MCP ツール管理（/v1/agent/chat 用）
# ============================================================
async def _load_mcp_tools():
    """mcp_clients.json を読み込んで MCP セッションを起動し、ツールスキーマを収集する"""
    global _mcp_sessions, _mcp_tools

    if not MCP_CLIENTS_CONFIG.exists():
        return

    try:
        config = json.loads(MCP_CLIENTS_CONFIG.read_text())
    except Exception as e:
        print(f"[mlx-proxy] mcp_clients.json 読み込みエラー: {e}", flush=True)
        return

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("[mlx-proxy] mcp パッケージが見つかりません。uv add mcp でインストールしてください。", flush=True)
        return

    servers = config.get("servers", [])
    new_sessions = {}
    new_tools = []

    for srv in servers:
        srv_name = srv.get("name", "unknown")
        try:
            if srv.get("transport") == "stdio":
                params = StdioServerParameters(
                    command=srv["command"],
                    args=srv.get("args", []),
                    env=srv.get("env"),
                )
                ctx = stdio_client(params)
                read, write = await ctx.__aenter__()
                session = ClientSession(read, write)
                await session.initialize()
                tool_list = await session.list_tools()
                for tool in tool_list.tools:
                    schema = {
                        "type": "function",
                        "function": {
                            "name": f"{srv_name}__{tool.name}",
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        },
                    }
                    new_tools.append(schema)
                    new_sessions[f"{srv_name}__{tool.name}"] = (session, srv_name, tool.name)
        except Exception as e:
            print(f"[mlx-proxy] MCP サーバ '{srv_name}' 接続エラー: {e}", flush=True)

    _mcp_sessions = new_sessions
    _mcp_tools = new_tools
    if new_tools:
        print(f"[mlx-proxy] MCP ツール {len(new_tools)} 件を登録しました", flush=True)


@app.get("/v1/agent/tools")
async def get_agent_tools():
    """登録済み MCP ツール一覧"""
    return {"tools": _mcp_tools, "count": len(_mcp_tools)}


# ============================================================
# API: エージェントループ（MCP ツール呼び出し込み）
# ============================================================
@app.post("/v1/agent/chat")
async def agent_chat(request: Request):
    """MCP ツールを呼び出すエージェントループ

    llm モードのみ対応。`tools` フィールドを指定しなくても、起動時にロードした
    MCP ツールが自動で付与される。
    事前に llm モードのモデルをロードしてください。
    """
    if not state["model_id"] or state["mode"] != "llm":
        raise HTTPException(
            status_code=409,
            detail="llm モードのモデルをロードしてください: POST /v1/models/load {\"mode\":\"llm\",...}"
        )

    payload = await request.json()
    messages = list(payload.get("messages", []))
    max_tokens = payload.get("max_tokens", 1024)
    temperature = payload.get("temperature", 0.0)

    # リクエストの tools と登録済み MCP ツールをマージ
    req_tools = payload.get("tools", [])
    tools = req_tools + _mcp_tools
    if not tools:
        # ツールなしなら普通の chat として転送
        return await chat_completions(request)

    tool_trace = []
    backend_url = f"http://127.0.0.1:{MLX_BACKEND_PORT}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=300) as client:
        for iteration in range(AGENT_MAX_ITERATIONS):
            req_body = {
                "model": state["model_id"],
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            r = await client.post(
                backend_url,
                content=json.dumps(req_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"バックエンドエラー: {r.text}")

            resp = r.json()
            choice = resp["choices"][0]
            finish_reason = choice.get("finish_reason")
            assistant_msg = choice["message"]
            messages.append(assistant_msg)

            # ツール呼び出しがなければ終了
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls or finish_reason == "stop":
                break

            # 各ツール呼び出しを MCP セッション経由で実行
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_full_name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                tool_result_content = ""
                if tool_full_name in _mcp_sessions:
                    session, srv_name, tool_name = _mcp_sessions[tool_full_name]
                    try:
                        call_result = await session.call_tool(tool_name, args)
                        tool_result_content = str(call_result.content) if call_result.content else ""
                    except Exception as e:
                        tool_result_content = f"ツール呼び出しエラー: {e}"
                else:
                    tool_result_content = f"未知のツール: {tool_full_name}"

                tool_trace.append({
                    "tool": tool_full_name,
                    "args": args,
                    "result": tool_result_content,
                    "iteration": iteration,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result_content,
                })
        else:
            # 最大反復に達した場合は打ち切り
            messages.append({
                "role": "assistant",
                "content": f"[エージェントループが最大反復回数 {AGENT_MAX_ITERATIONS} に達したため打ち切り]",
            })

    final_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"),
        {"role": "assistant", "content": ""},
    )

    return JSONResponse(content={
        "id": f"agent-{uuid.uuid4().hex[:8]}",
        "object": "agent.chat.completion",
        "created": int(time.time()),
        "model": state["model_id"],
        "choices": [{
            "index": 0,
            "message": final_assistant,
            "finish_reason": "stop",
        }],
        "tool_trace": tool_trace,
    })


# ============================================================
# ヘルスチェック
# ============================================================
@app.get("/health")
async def health():
    if state["mode"] == "vlm":
        backend_ok = state["vlm_model"] is not None
    elif state["mode"] == "score":
        backend_ok = state["score_model"] is not None
    else:
        backend_ok = False
        if state["process"] and state["process"].poll() is None:
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    r = await client.get(f"http://127.0.0.1:{MLX_BACKEND_PORT}/v1/models")
                    backend_ok = r.status_code == 200
            except Exception:
                pass
    result = {
        "proxy": "ok",
        "backend": "ok" if backend_ok else "not_running",
        "model_id": state["model_id"],
        "enable_thinking": state["enable_thinking"],
        "mode": state["mode"],
        "mcp_tools": len(_mcp_tools),
    }
    if state["vlm_draft_model"] is not None:
        result["draft_kind"] = state["vlm_draft_kind"]
    return result


# ============================================================
# ライフサイクル
# ============================================================
@app.on_event("startup")
async def startup():
    """起動時に MCP ツールをロード"""
    await _load_mcp_tools()


@app.on_event("shutdown")
async def shutdown():
    """シャットダウン時にバックエンドとMCPセッションを落とす"""
    # MCP セッションを閉じる
    closed = set()
    for key, (session, srv_name, tool_name) in _mcp_sessions.items():
        if srv_name not in closed:
            try:
                await session.aclose()
            except Exception:
                pass
            closed.add(srv_name)
    await kill_backend()


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
