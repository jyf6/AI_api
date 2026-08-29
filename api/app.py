from __future__ import annotations

import base64
import asyncio
import json
import time
from typing import Literal
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI
from services.doubao_account_service import doubao_account_service
from services.doubao_backend_api import DoubaoBackendAPI
from services.gemini_account_service import gemini_account_service
from services.gemini_backend_api import GeminiBackendAPI
from utils.helper import UpstreamHTTPError
from utils.log import logger


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="生成图片的提示词")
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1)
    size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    response_format: Literal["b64_json", "url"] = "b64_json"
    image: str | list[str] | None = None


class OAuthCallbackRequest(BaseModel):
    callback_url: str = Field(..., min_length=1, description="浏览器授权后跳转的完整 URL 或 Code")
    session_id: str = Field(default="", description="OAuth 会话 ID")
    proxy: str = Field(default="", description="可选独立代理节点 (http://user:pass@ip:port 或 socks5://...)")


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-5-5"
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    stream: bool = False


class DoubaoAccountRequest(BaseModel):
    name: str = ""
    cookie: str = Field(..., min_length=1)
    proxy: str = ""


class GeminiAccountRequest(BaseModel):
    name: str = ""
    cookie: str = Field(..., min_length=1)
    proxy: str = ""


def _size_to_ratio(size: str) -> str:
    return {"1792x1024": "16:9", "1024x1792": "9:16"}.get(size, "1:1")


def create_app() -> FastAPI:
    app = FastAPI(title="ChatGPT-Image-Service", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. 核心文生图接口 (OpenAI 兼容规范)
    @app.post("/v1/images/generations")
    async def generate_images(body: ImageGenerationRequest):
        if body.model.startswith("gemini-"):
            try:
                account = gemini_account_service.get_available_account()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            try:
                async with GeminiBackendAPI(account) as backend:
                    refs = [body.image] if isinstance(body.image, str) else (body.image or [])
                    images = await backend.image(body.prompt, body.model, refs)
                gemini_account_service.release_account(account["name"], True)
                return {"created": int(time.time()), "data": [
                    {"b64_json": base64.b64encode(image).decode("ascii")} for image in images
                ]}
            except Exception as exc:
                if "account" in locals():
                    gemini_account_service.release_account(account["name"], False, str(exc))
                raise HTTPException(status_code=502, detail=f"Gemini image generation failed: {exc}") from exc
        if body.model.startswith("doubao-"):
            try:
                account = doubao_account_service.get_available_account()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            try:
                async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
                    urls = await backend.generate_image(body.prompt, ratio=_size_to_ratio(body.size))
                    images = await backend.download_images(urls)
                doubao_account_service.release_account(account["name"], True)
                return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(image).decode("ascii")} for image in images]}
            except Exception as exc:
                doubao_account_service.release_account(account["name"], False, str(exc))
                raise HTTPException(status_code=502, detail=f"Doubao image generation failed: {exc}") from exc
        start_time = time.time()
        def _do_generate(token: str, proxy: str) -> bytes | list[bytes]:
            with OpenAIBackendAPI(access_token=token, proxy=proxy) as backend:
                return backend.generate_image_bytes(body.prompt, body.model)

        async def generate_one() -> list[bytes]:
            try:
                account = account_service.get_available_account()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"No available account in pool: {exc}")
            email = account["email"]
            proxy = account.get("proxy", "")
            try:
                image = await asyncio.to_thread(_do_generate, account["access_token"], proxy)
                account_service.release_account(email, success=True)
                return image if isinstance(image, list) else [image]
            except asyncio.CancelledError:
                account_service.release_account(email, success=False, error_msg="request cancelled")
                raise
            except Exception as exc:
                status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
                retry_after = exc.retry_after if isinstance(exc, UpstreamHTTPError) else None
                error = str(exc)
                account_service.release_account(email, success=False, error_msg=error,
                                                status_code=status, retry_after=retry_after)
                logger.warning(f"Image generation failed on {email}: {error}")
                raise HTTPException(status_code=502, detail=f"Image generation failed: {error}") from exc

        images = await generate_one()

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Image generation succeeded in {elapsed}s")

        data = []
        for image in images:
            b64_data = base64.b64encode(image).decode("ascii")
            data.append({"b64_json": b64_data} if body.response_format == "b64_json" else {
                "url": f"data:image/png;base64,{b64_data}"
            })
        return {
            "created": int(time.time()),
            "data": data,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        """网页 ChatGPT 文本/多模态代理；仅返回分析文本，不串联生图。"""
        if body.model.startswith("gemini-"):
            if body.stream:
                raise HTTPException(status_code=400, detail="Gemini Web chat currently supports stream=false only")
            try:
                account = gemini_account_service.get_available_account()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            try:
                async with GeminiBackendAPI(account) as backend:
                    text = await backend.chat(body.messages, body.model)
                gemini_account_service.release_account(account["name"], True)
                return {"id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                        "model": body.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
            except Exception as exc:
                if "account" in locals():
                    gemini_account_service.release_account(account["name"], False, str(exc))
                raise HTTPException(status_code=502, detail=f"Gemini chat failed: {exc}") from exc
        if body.model.startswith("doubao-"):
            text_parts = []
            image_values = []
            for message in body.messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif part.get("type") in {"image_url", "image"}:
                            image = part.get("image_url")
                            image_values.append(image.get("url") if isinstance(image, dict) else part.get("data"))
            prompt = "\n".join(part for part in text_parts if part)
            if not prompt and not image_values:
                raise HTTPException(status_code=400, detail="Doubao chat requires text or image content")
            try:
                account = doubao_account_service.get_available_account()
                async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
                    attachments = [await backend.upload_image(str(value)) for value in image_values if value]
                    text = await backend.chat(prompt, attachments)
                doubao_account_service.release_account(account["name"], True)
                return {"id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                        "model": body.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
            except Exception as exc:
                if "account" in locals():
                    doubao_account_service.release_account(account["name"], False, str(exc))
                raise HTTPException(status_code=502, detail=f"Doubao chat failed: {exc}") from exc
        try:
            account = account_service.get_available_account()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"No available account in pool: {exc}")
        email, proxy = account["email"], account.get("proxy", "")

        def run() -> str:
            text = ""
            with OpenAIBackendAPI(account["access_token"], proxy) as backend:
                for raw in backend.stream_chat(body.messages, body.model):
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except Exception:
                        continue
                    candidates = [event]
                    nested = event.get("v") if isinstance(event, dict) else None
                    if isinstance(nested, str):
                        try:
                            nested = json.loads(nested)
                        except Exception:
                            nested = None
                    if isinstance(nested, dict):
                        candidates.append(nested)
                    for candidate in candidates:
                        message = candidate.get("message") if isinstance(candidate, dict) else None
                        if not isinstance(message, dict) or (message.get("author") or {}).get("role") != "assistant":
                            continue
                        content = message.get("content") or {}
                        parts = content.get("parts") or []
                        if isinstance(content.get("text"), str):
                            text = content["text"]
                        elif parts:
                            text = "".join(str(part) for part in parts if isinstance(part, str))
            if not text.strip():
                raise RuntimeError("Upstream chat returned no assistant text")
            return text

        try:
            text = await asyncio.to_thread(run)
            account_service.release_account(email, success=True)
        except asyncio.CancelledError:
            account_service.release_account(email, success=False, error_msg="request cancelled")
            raise
        except Exception as exc:
            error = str(exc)
            status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
            retry_after = exc.retry_after if isinstance(exc, UpstreamHTTPError) else None
            account_service.release_account(email, success=False, error_msg=error, status_code=status, retry_after=retry_after)
            raise HTTPException(status_code=502, detail=f"Chat completion failed: {error}") from exc
        return {"id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                "model": body.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

    # 2. 浏览器 OAuth 授权登录接口
    @app.post("/api/oauth/authorize")
    async def start_oauth():
        """生成 OAuth 授权链接"""
        return account_service.start_oauth_session()

    @app.post("/api/oauth/callback")
    async def finish_oauth(body: OAuthCallbackRequest):
        """用浏览器授权跳转的 URL 换取 Token 并自动录入号池"""
        try:
            account = account_service.finish_oauth_session(body.callback_url, body.proxy, body.session_id)
            return {"code": 0, "message": "Account added successfully", "data": account}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 3. 号池管理接口 (RESTful)
    @app.get("/api/accounts")
    async def list_accounts():
        return {"accounts": account_service.list_accounts()}

    @app.delete("/api/accounts/{email}")
    async def delete_account(email: str):
        success = account_service.delete_account(email)
        if not success:
            raise HTTPException(status_code=404, detail="Account not found")
        return {"code": 0, "message": f"Account {email} deleted"}

    @app.post("/api/accounts/{email}/refresh")
    async def refresh_account(email: str):
        try:
            acc = account_service.refresh_account(email)
            return {"code": 0, "message": f"Account {email} refreshed", "data": acc["email"]}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/stats")
    async def get_stats():
        return account_service.get_stats()

    @app.get("/api/doubao/accounts")
    async def list_doubao_accounts():
        return {"accounts": doubao_account_service.list_accounts()}

    @app.post("/api/doubao/accounts")
    async def add_doubao_account(body: DoubaoAccountRequest):
        try:
            account = doubao_account_service.add_account(body.name, body.cookie, body.proxy)
            return {"name": account["name"], "status": account["status"]}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/doubao/accounts/{name}")
    async def delete_doubao_account(name: str):
        if not doubao_account_service.delete_account(name):
            raise HTTPException(status_code=404, detail="Doubao account not found")
        return {"status": "deleted"}

    @app.get("/api/doubao/stats")
    async def doubao_stats():
        return doubao_account_service.stats()

    @app.get("/api/gemini/accounts")
    async def list_gemini_accounts():
        return {"accounts": gemini_account_service.list_accounts()}

    @app.post("/api/gemini/accounts")
    async def add_gemini_account(body: GeminiAccountRequest):
        try:
            account = gemini_account_service.add_account(body.name, body.cookie, body.proxy)
            return {"name": account["name"], "status": account["status"]}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/gemini/accounts/{name}")
    async def delete_gemini_account(name: str):
        if not gemini_account_service.delete_account(name):
            raise HTTPException(status_code=404, detail="Gemini account not found")
        return {"status": "deleted"}

    @app.get("/api/gemini/stats")
    async def gemini_stats():
        return gemini_account_service.stats()

    @app.get("/health")
    async def health_check():
        stats = account_service.get_stats()
        healthy = stats["active_accounts"] > 0
        return {"status": "ok" if healthy else "degraded", "healthy": healthy, "timestamp": int(time.time()), "accounts": stats}

    # 4. 内置单页可视化调试面板 (支持浏览器 OAuth 录入 + 文生图实时预览)
    @app.get("/", response_class=HTMLResponse)
    async def debug_ui():
        return HTML_DASHBOARD_PAGE

    return app


HTML_DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ChatGPT 生图微服务 - 控制台与调试台</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --success: #4ade80;
      --danger: #f87171;
      --warning: #facc15;
      --input-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; }
    .container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }
    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--card-border); padding-bottom: 16px; }
    .header h1 { font-size: 22px; font-weight: 700; color: var(--primary); }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-success { background: rgba(74, 222, 128, 0.15); color: var(--success); }
    .badge-danger { background: rgba(248, 113, 113, 0.15); color: var(--danger); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 20px; }
    .card-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    .form-group { margin-bottom: 14px; }
    .form-group label { display: block; font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }
    .form-control { width: 100%; padding: 10px; background: var(--input-bg); border: 1px solid var(--card-border); border-radius: 6px; color: var(--text); font-size: 14px; }
    .form-control:focus { outline: none; border-color: var(--primary); }
    textarea.form-control { resize: vertical; min-height: 90px; }
    .btn { padding: 9px 16px; border-radius: 6px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; justify-content: center; gap: 6px; }
    .btn-primary { background: var(--primary); color: #000; }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-success { background: var(--success); color: #000; }
    .btn-success:hover { opacity: 0.9; }
    .btn-danger { background: rgba(248, 113, 113, 0.2); color: var(--danger); }
    .btn-danger:hover { background: var(--danger); color: #fff; }
    .btn-sm { padding: 4px 8px; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--card-border); }
    th { color: var(--text-muted); font-weight: 600; }
    .preview-box { min-height: 260px; background: var(--input-bg); border: 2px dashed var(--card-border); border-radius: 6px; display: flex; justify-content: center; align-items: center; flex-direction: column; overflow: hidden; margin-top: 14px; }
    .preview-box img { max-width: 100%; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    .stats-row { display: flex; gap: 12px; margin-bottom: 16px; }
    .stat-pill { background: var(--input-bg); border: 1px solid var(--card-border); padding: 8px 14px; border-radius: 6px; font-size: 13px; }
    .stat-pill strong { color: var(--primary); }
    .loading { display: none; color: var(--primary); font-size: 14px; margin-top: 10px; }
    .oauth-box { background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 12px; }
    .step-tag { background: var(--card-border); color: var(--primary); font-size: 12px; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
    .multimodal-drop { min-height: 150px; border: 2px dashed var(--card-border); border-radius: 6px; background: var(--input-bg); display: flex; align-items: center; justify-content: center; text-align: center; padding: 16px; cursor: pointer; }
    .multimodal-drop.has-image { border-style: solid; }
    .multimodal-drop img { max-width: 100%; max-height: 220px; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ChatGPT 生图微服务控制台</h1>
      <div><span class="badge badge-success">API 运行中</span></div>
    </div>

    <!-- 号池概览与 OAuth 录入 -->
    <div class="card">
      <div class="card-title">
        <span>号池状态与账号录入</span>
        <button class="btn btn-sm btn-primary" onclick="fetchAccounts()">刷新号池</button>
      </div>
      <div class="stats-row">
        <div class="stat-pill">总账号数: <strong id="stat-total">0</strong></div>
        <div class="stat-pill">可用账号: <strong id="stat-active">0</strong></div>
        <div class="stat-pill">正在生图: <strong id="stat-inflight">0</strong></div>
      </div>

      <!-- 浏览器 OAuth 授权录入流程卡片 -->
      <div class="oauth-box">
        <div style="font-size: 14px; font-weight: 600; color: #fff; display: flex; align-items: center; gap: 8px;">
          <span>🌐 快捷录入：浏览器授权登录 (无需手动获取 Token)</span>
        </div>
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
          <button type="button" class="btn btn-primary" onclick="openOAuthWindow()">
            <span>1. 打开浏览器登录</span>
          </button>
          <span style="font-size: 13px; color: var(--text-muted);">
            点击在新标签页登录 OpenAI，登录完成后将浏览器地址栏跳转的 <strong>完整 URL</strong> 复制并粘贴到下方。
          </span>
        </div>

        <form id="callback-form" onsubmit="handleCallbackSubmit(event)" style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px;">
          <input type="text" id="oauth-callback-url" class="form-control" placeholder="2. 粘贴浏览器地址栏重定向后的完整 URL (https://platform.openai.com/auth/callback?code=...)" required style="flex: 2; min-width: 280px;">
          <input type="text" id="oauth-proxy" class="form-control" placeholder="3. 节点代理 (选填: http://ip:port 或 socks5://)" style="flex: 1.2; min-width: 200px;">
          <button type="submit" id="btn-submit-oauth" class="btn btn-success" style="white-space: nowrap;">4. 确认录入账号</button>
        </form>
      </div>

      <table>
        <thead>
          <tr>
            <th>邮箱</th>
            <th>代理节点</th>
            <th>计划</th>
            <th>状态</th>
            <th>在途</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody id="account-table-body">
          <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">暂无账号，请在上方录入</td></tr>
        </tbody>
      </table>
    </div>

    <!-- 豆包 Cookie 账号池 -->
    <div class="card">
      <div class="card-title"><span>豆包 Cookie 账号池</span><button class="btn btn-sm btn-primary" onclick="fetchDoubaoAccounts()">刷新</button></div>
      <form onsubmit="handleDoubaoAccount(event)" style="display:flex;gap:10px;flex-wrap:wrap;">
        <input id="doubao-name" class="form-control" placeholder="账号名称" style="flex:1;min-width:150px;">
        <input id="doubao-cookie" class="form-control" placeholder="粘贴豆包完整 Cookie Header" required style="flex:3;min-width:300px;">
        <input id="doubao-proxy" class="form-control" placeholder="代理（选填）" style="flex:1.5;min-width:180px;">
        <button class="btn btn-success" type="submit">录入豆包账号</button>
      </form>
      <table><thead><tr><th>名称</th><th>状态</th><th>并发</th><th>操作</th></tr></thead><tbody id="doubao-table-body"><tr><td colspan="4" style="text-align:center;color:var(--text-muted);">暂无豆包账号</td></tr></tbody></table>
    </div>

    <div class="card">
      <div class="card-title"><span>Gemini Web Cookie 账号</span><button class="btn btn-sm btn-primary" onclick="fetchGeminiAccounts()">刷新</button></div>
      <form onsubmit="handleGeminiAccount(event)" style="display:flex;gap:10px;flex-wrap:wrap;">
        <input id="gemini-name" class="form-control" placeholder="账号名称" style="flex:1;min-width:150px;">
        <input id="gemini-cookie" class="form-control" placeholder="粘贴 Gemini 完整 Cookie（含 __Secure-1PSID 与 __Secure-1PSIDTS）" required style="flex:3;min-width:300px;">
        <input id="gemini-proxy" class="form-control" placeholder="代理（选填）" style="flex:1.5;min-width:180px;">
        <button class="btn btn-success" type="submit">录入 Gemini 账号</button>
      </form>
      <table><thead><tr><th>名称</th><th>状态</th><th>并发</th><th>失败次数</th><th>操作</th></tr></thead><tbody id="gemini-table-body"><tr><td colspan="5" style="text-align:center;color:var(--text-muted);">暂无 Gemini 账号</td></tr></tbody></table>
    </div>

    <!-- 网页 ChatGPT 多模态分析调试台 -->
    <div class="card">
      <div class="card-title">网页 ChatGPT 多模态分析</div>
      <div class="grid">
        <div>
          <div class="form-group">
            <label>分析文本:</label>
            <textarea id="chat-prompt" class="form-control" placeholder="例如：分析图片中的主体、风格、构图和配色"></textarea>
          </div>
          <div class="form-group">
            <label>参考图片:</label>
            <input id="chat-image-file" type="file" accept="image/*" style="display:none" onchange="handleChatImageFile(event)">
            <div id="chat-image-drop" class="multimodal-drop" onclick="document.getElementById('chat-image-file').click()" onpaste="handleChatPaste(event)">
              <span>点击选择图片，或先点击此区域后按 Ctrl+V 粘贴图片</span>
            </div>
          </div>
          <div style="display: flex; gap: 12px; align-items: end;">
            <div class="form-group" style="flex: 1; margin-bottom: 0;">
              <label>多模态聊天模型:</label>
              <select id="chat-model" class="form-control">
                <option value="gpt-5-5">GPT 网页多模态 (gpt-5-5)</option>
                <option value="doubao-chat">豆包网页多模态 (doubao-chat)</option>
                <option value="gemini-3-flash">Gemini 网页多模态 (gemini-3-flash)</option>
              </select>
            </div>
            <button id="btn-chat" class="btn btn-primary" onclick="handleChatCompletion()">调用模型分析</button>
          </div>
          <div id="chat-status" class="loading">正在调用网页 ChatGPT，请耐心等待...</div>
        </div>
        <div>
          <label style="font-size: 13px; color: var(--text-muted); display: block; margin-bottom: 6px;">分析结果:</label>
          <div id="chat-result" class="preview-box" style="justify-content: flex-start; align-items: flex-start; padding: 14px; white-space: pre-wrap; text-align: left;"><span style="color: var(--text-muted); font-size: 13px;">等待调用...</span></div>
        </div>
      </div>
    </div>

    <!-- 在线文生图调试台 -->
    <div class="card">
      <div class="card-title">文生图调试台 (OpenAI /v1/images/generations)</div>
      <div class="grid">
        <div>
          <div class="form-group">
            <label>提示词 (Prompt):</label>
            <textarea id="gen-prompt" class="form-control" placeholder="输入生图 Prompt，例如: A cute cyberpunk cat on a neon street, 8k resolution"></textarea>
          </div>
          <div class="form-group">
            <label>Gemini 参考图（选填）:</label>
            <input id="gen-reference-image" type="file" accept="image/*" class="form-control">
          </div>
          <div style="display: flex; gap: 12px;">
            <div class="form-group" style="flex: 1;">
              <label>模型 (Model):</label>
              <select id="gen-model" class="form-control">
                <option value="gpt-image-2">gpt-image-2 (推荐)</option>
                <option value="doubao-image">doubao-image</option>
                <option value="gemini-2.5-pro-image">Gemini 生图 (gemini-2.5-pro-image)</option>
              </select>
            </div>
            <div class="form-group" style="flex: 1;">
              <label>尺寸 (Size):</label>
              <select id="gen-size" class="form-control">
                <option value="1024x1024">1024x1024 (正方形)</option>
                <option value="1792x1024">1792x1024 (横屏 16:9)</option>
                <option value="1024x1792">1024x1792 (竖屏 9:16)</option>
              </select>
            </div>
          </div>
          <button id="btn-generate" class="btn btn-primary" style="width: 100%; margin-top: 8px;" onclick="handleGenerate()">立即生成图片</button>
          <div id="gen-status" class="loading">正在逆向握手并生成中，请耐心等待 (约 15-30s)...</div>
        </div>

        <div>
          <label style="font-size: 13px; color: var(--text-muted); display: block; margin-bottom: 6px;">生成结果预览:</label>
          <div id="preview-container" class="preview-box">
            <span style="color: var(--text-muted); font-size: 13px;">等待生成...</span>
          </div>
          <div id="gen-meta" style="margin-top: 10px; font-size: 12px; color: var(--text-muted);"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentSessionId = "";
    let chatImageData = "";
    let generationReferenceImage = "";

    document.getElementById('gen-reference-image').addEventListener('change', event => {
      const file = event.target.files[0];
      if (!file) { generationReferenceImage = ""; return; }
      const reader = new FileReader();
      reader.onload = () => generationReferenceImage = reader.result;
      reader.readAsDataURL(file);
    });

    async function fetchDoubaoAccounts() {
      const res = await fetch('/api/doubao/accounts');
      const data = await res.json();
      const body = document.getElementById('doubao-table-body');
      body.innerHTML = (data.accounts || []).map(acc => `<tr><td>${acc.name}</td><td><span class="badge ${acc.status === 'active' ? 'badge-success' : 'badge-danger'}">${acc.status}</span></td><td>${acc.inflight || 0}</td><td><button class="btn btn-sm btn-danger" onclick="deleteDoubaoAccount('${encodeURIComponent(acc.name)}')">删除</button></td></tr>`).join('') || '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">暂无豆包账号</td></tr>';
    }

    async function handleDoubaoAccount(event) {
      event.preventDefault();
      const res = await fetch('/api/doubao/accounts', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:document.getElementById('doubao-name').value, cookie:document.getElementById('doubao-cookie').value, proxy:document.getElementById('doubao-proxy').value})});
      const data = await res.json();
      if (!res.ok) return alert(data.detail || '豆包账号录入失败');
      event.target.reset();
      fetchDoubaoAccounts();
    }

    async function deleteDoubaoAccount(name) {
      await fetch('/api/doubao/accounts/' + name, {method:'DELETE'});
      fetchDoubaoAccounts();
    }

    async function fetchGeminiAccounts() {
      const res = await fetch('/api/gemini/accounts');
      const data = await res.json();
      const body = document.getElementById('gemini-table-body');
      body.innerHTML = (data.accounts || []).map(acc => `<tr><td>${acc.name}</td><td><span class="badge ${acc.status === 'active' ? 'badge-success' : 'badge-danger'}">${acc.status}</span></td><td>${acc.inflight || 0}</td><td>${acc.failure_count || 0}</td><td><button class="btn btn-sm btn-danger" onclick="deleteGeminiAccount('${encodeURIComponent(acc.name)}')">删除</button></td></tr>`).join('') || '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">暂无 Gemini 账号</td></tr>';
    }

    async function handleGeminiAccount(event) {
      event.preventDefault();
      const res = await fetch('/api/gemini/accounts', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:document.getElementById('gemini-name').value, cookie:document.getElementById('gemini-cookie').value, proxy:document.getElementById('gemini-proxy').value})});
      const data = await res.json();
      if (!res.ok) return alert(data.detail || 'Gemini 账号录入失败');
      event.target.reset();
      fetchGeminiAccounts();
    }

    async function deleteGeminiAccount(name) {
      await fetch('/api/gemini/accounts/' + name, {method:'DELETE'});
      fetchGeminiAccounts();
    }

    function setChatImage(file) {
      if (!file || !file.type.startsWith('image/')) return alert('请选择图片文件');
      const reader = new FileReader();
      reader.onload = () => {
        chatImageData = reader.result;
        const drop = document.getElementById('chat-image-drop');
        drop.classList.add('has-image');
        drop.innerHTML = `<img src="${chatImageData}" alt="待分析图片" />`;
      };
      reader.readAsDataURL(file);
    }

    function handleChatImageFile(event) {
      setChatImage(event.target.files[0]);
    }

    function handleChatPaste(event) {
      const item = Array.from(event.clipboardData?.items || []).find(item => item.type.startsWith('image/'));
      if (item) {
        event.preventDefault();
        setChatImage(item.getAsFile());
      }
    }

    async function handleChatCompletion() {
      const prompt = document.getElementById('chat-prompt').value.trim();
      const model = document.getElementById('chat-model').value.trim() || 'gpt-5-5';
      if (!prompt) return alert('请输入分析文本');
      if (!chatImageData) return alert('请选择或粘贴一张图片');

      const btn = document.getElementById('btn-chat');
      const status = document.getElementById('chat-status');
      const result = document.getElementById('chat-result');
      btn.disabled = true;
      status.style.display = 'block';
      result.innerText = '正在分析...';
      try {
        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({model, messages: [{role: 'user', content: [
            {type: 'text', text: prompt},
            {type: 'image_url', image_url: {url: chatImageData}}
          ]}]})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '多模态分析失败');
        result.innerText = data.choices?.[0]?.message?.content || '模型未返回文本';
      } catch (err) {
        result.innerText = '分析失败: ' + err.message;
      } finally {
        btn.disabled = false;
        status.style.display = 'none';
        fetchAccounts();
      }
    }

    async function fetchAccounts() {
      try {
        const [accRes, statsRes] = await Promise.all([
          fetch('/api/accounts').then(r => r.json()),
          fetch('/api/stats').then(r => r.json())
        ]);
        document.getElementById('stat-total').innerText = statsRes.total_accounts || 0;
        document.getElementById('stat-active').innerText = statsRes.active_accounts || 0;
        document.getElementById('stat-inflight').innerText = statsRes.total_inflight_tasks || 0;

        const tbody = document.getElementById('account-table-body');
        if (!accRes.accounts || accRes.accounts.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted);">号池为空，请在上方添加</td></tr>';
          return;
        }
        tbody.innerHTML = accRes.accounts.map(acc => `
          <tr>
            <td><strong>${acc.email}</strong></td>
            <td><code>${acc.proxy || '宿主直连'}</code></td>
            <td>${acc.plan_type || '-'}</td>
            <td><span class="badge ${acc.status === 'active' ? 'badge-success' : 'badge-danger'}">${acc.status}</span></td>
            <td>${acc.image_inflight || 0}</td>
            <td>
              <button class="btn btn-sm btn-primary" onclick="handleRefresh('${acc.email}')">刷新</button>
              <button class="btn btn-sm btn-danger" onclick="handleDelete('${acc.email}')">删除</button>
            </td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    async function openOAuthWindow() {
      try {
        const res = await fetch('/api/oauth/authorize', {method: 'POST'});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '生成授权链接失败');
        currentSessionId = data.session_id;
        window.open(data.authorize_url, '_blank');
      } catch (err) {
        alert('生成授权链接失败: ' + err.message);
      }
    }

    async function handleCallbackSubmit(e) {
      e.preventDefault();
      const callback_url = document.getElementById('oauth-callback-url').value.trim();
      const proxy = document.getElementById('oauth-proxy').value.trim();
      const btn = document.getElementById('btn-submit-oauth');

      btn.disabled = true;
      btn.innerText = '兑换中...';

      try {
        const res = await fetch('/api/oauth/callback', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({callback_url, proxy, session_id: currentSessionId})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '录入失败');
        alert('🎉 账号录入成功！邮箱: ' + data.data.email);
        document.getElementById('callback-form').reset();
        fetchAccounts();
      } catch (err) {
        alert('录入失败: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerText = '4. 确认录入账号';
      }
    }

    async function handleDelete(email) {
      if (!confirm('确定删除账号 ' + email + ' 吗？')) return;
      await fetch('/api/accounts/' + encodeURIComponent(email), {method: 'DELETE'});
      fetchAccounts();
    }

    async function handleRefresh(email) {
      try {
        const res = await fetch('/api/accounts/' + encodeURIComponent(email) + '/refresh', {method: 'POST'});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '刷新失败');
        alert('Token 刷新成功！');
        fetchAccounts();
      } catch (err) {
        alert('刷新失败: ' + err.message);
      }
    }

    async function handleGenerate() {
      const prompt = document.getElementById('gen-prompt').value.trim();
      if (!prompt) return alert('请输入提示词');
      const model = document.getElementById('gen-model').value;
      const size = document.getElementById('gen-size').value;

      const btn = document.getElementById('btn-generate');
      const status = document.getElementById('gen-status');
      const container = document.getElementById('preview-container');
      const meta = document.getElementById('gen-meta');

      btn.disabled = true;
      status.style.display = 'block';
      container.innerHTML = '<span style="color: var(--primary);">正在生成中...</span>';
      meta.innerText = '';

      const t0 = performance.now();
      try {
        const res = await fetch('/v1/images/generations', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({prompt, model, size, response_format: 'b64_json', image: generationReferenceImage || undefined})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '生图失败');

        const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
        const b64 = data.data[0].b64_json;
        container.innerHTML = `<img src="data:image/png;base64,${b64}" alt="Generated image" />`;
        meta.innerHTML = `耗时: <strong>${elapsed}s</strong> | 格式: PNG (Base64) | <a href="data:image/png;base64,${b64}" download="image.png" style="color: var(--primary);">点击下载原图</a>`;
      } catch (err) {
        container.innerHTML = `<span style="color: var(--danger);">生成失败: ${err.message}</span>`;
      } finally {
        btn.disabled = false;
        status.style.display = 'none';
        fetchAccounts();
      }
    }

    // 初始化加载
    fetchAccounts();
    fetchDoubaoAccounts();
    fetchGeminiAccounts();
  </script>
</body>
</html>
"""
