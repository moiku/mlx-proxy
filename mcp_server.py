#!/usr/bin/env python3
"""
mcp_server.py - mlx-proxy の MCP サーバ

mlx-proxy (localhost:8080) を MCP ツールとして公開する薄いラッパ。
FastAPI アプリとは独立した別プロセスとして起動する。

トランスポート:
  stdio  (default) - Claude Code 等のローカル MCP クライアント向け
  http             - Tailscale 経由のリモート MCP クライアント向け（:8090）

起動方法:
  # stdio（Claude Code から自動起動）
  uv run --project ~/Services/mlx-proxy python ~/Services/mlx-proxy/mcp_server.py

  # HTTP（launchd または手動）
  MCP_TRANSPORT=http MCP_PORT=8090 uv run --project ~/Services/mlx-proxy python ~/Services/mlx-proxy/mcp_server.py

Claude Code への登録:
  claude mcp add mlx-proxy -- uv run --project /Users/gendo/Services/mlx-proxy python /Users/gendo/Services/mlx-proxy/mcp_server.py

または .mcp.json:
  {
    "mcpServers": {
      "mlx-proxy": {
        "command": "uv",
        "args": ["run", "--project", "/Users/gendo/Services/mlx-proxy",
                 "python", "/Users/gendo/Services/mlx-proxy/mcp_server.py"]
      }
    }
  }
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP

PROXY_URL = os.environ.get("MLX_PROXY_URL", "http://localhost:8080")
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
MCP_PORT = int(os.environ.get("MCP_PORT", "8090"))

mcp = FastMCP("mlx-proxy")


# ============================================================
# ツール定義
# ============================================================

@mcp.tool()
async def health() -> dict:
    """mlx-proxy とバックエンドの状態を確認する"""
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(f"{PROXY_URL}/health")
        return r.json()


@mcp.tool()
async def list_models() -> dict:
    """利用可能なモデル一覧を取得する（ロード済みのものがわかる）"""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{PROXY_URL}/v1/models")
        return r.json()


@mcp.tool()
async def load_model(
    model_id: str,
    mode: str = "llm",
    enable_thinking: bool = False,
) -> dict:
    """モデルをロードする

    Args:
        model_id: モデル ID またはパス（例: "mlx-community/Qwen3-8B-4bit"）
        mode: "llm"（テキスト生成）/ "vlm"（画像+テキスト）/ "score"（logprob 直接取得）
        enable_thinking: Qwen3 等の thinking モードを有効にするか（llm モードのみ有効）
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{PROXY_URL}/v1/models/load",
            json={"model_id": model_id, "mode": mode, "enable_thinking": enable_thinking},
        )
        return r.json()


@mcp.tool()
async def unload_model() -> dict:
    """現在ロード中のモデルをアンロードし GPU メモリを解放する"""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{PROXY_URL}/v1/models/unload")
        return r.json()


@mcp.tool()
async def chat(
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    stream: bool = False,
) -> dict:
    """ロード済みモデルとチャットする（llm / vlm モード）

    Args:
        messages: OpenAI 形式のメッセージリスト
                  例: [{"role": "user", "content": "こんにちは"}]
        max_tokens: 最大生成トークン数
        temperature: サンプリング温度（0 で決定論的）
        stream: ストリーミング（プロキシ経由では非推奨; 直接 :18080 を使用すること）
    """
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            },
        )
        return r.json()


@mcp.tool()
async def score(
    candidates: list,
    prompt: str = "",
    messages: list = None,
    apply_chat_template: bool = False,
    multi_token: str = "first",
) -> dict:
    """指定トークンの logprob を直接返す（score モードが必要）

    top-N 取りこぼし問題を根絶する。OpenAI 互換サーバの top_logprobs では
    指定トークンが top-N に入らないと取りこぼすが、このエンドポイントは
    全 logits から直接指定トークンの確率を計算する。

    事前に load_model(mode="score") でモデルをロードしてください。

    Args:
        candidates: スコアリングするトークン文字列のリスト（例: ["1","2","3","4","5"]）
        prompt: 生文字列プロンプト（apply_chat_template=False の場合）
        messages: chat messages（apply_chat_template=True の場合）
        apply_chat_template: True の場合 messages を chat template に通す
        multi_token: "first"（高速・最初のトークンのみ）または "sum"（teacher-forcing）

    Returns:
        各候補の logprob と prob_normalized（候補集合内 softmax）、argmax
    """
    payload: dict = {
        "candidates": candidates,
        "apply_chat_template": apply_chat_template,
        "multi_token": multi_token,
    }
    if apply_chat_template and messages:
        payload["messages"] = messages
    else:
        payload["prompt"] = prompt

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{PROXY_URL}/v1/score", json=payload)
        return r.json()


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    if MCP_TRANSPORT == "http":
        # streamable-http: Tailscale 経由のリモート MCP クライアント向け
        mcp.run(transport="streamable-http", host="0.0.0.0", port=MCP_PORT)
    else:
        # stdio: ローカル Claude Code 向け（デフォルト）
        mcp.run(transport="stdio")
