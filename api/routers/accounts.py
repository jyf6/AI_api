from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import DoubaoAccountRequest, GeminiAccountRequest, ModelConfigUpdateRequest
from core.database import database
from core.router import resolve_model
from providers.openai.account import account_service
from providers.doubao.account import doubao_account_service
from providers.gemini.account import gemini_account_service
from providers.openai.backend import OpenAIBackendAPI
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.backend import GeminiBackendAPI
import time
import asyncio
from typing import Any

router = APIRouter(tags=["accounts"])


def _model_capability_view(platform: str, pool: Any) -> dict[str, Any]:
    key = "email" if platform == "gpt" else "name"
    accounts = pool.list_accounts()
    active = [account for account in accounts if account.get("status") == "active"]
    model_sets = [
        {str(model.get("value")) for model in account.get("supported_models", []) if model.get("value")}
        for account in active
    ]
    complete = bool(active) and all(model_sets)
    intersection = sorted(set.intersection(*model_sets)) if complete else []
    return {
        "platform": platform,
        "accounts": [
            {
                "account": account.get(key, ""),
                "status": account.get("status", ""),
                "models": account.get("supported_models", []),
                "models_updated_at": account.get("models_updated_at"),
            }
            for account in accounts
        ],
        "intersection": intersection,
        "intersection_complete": complete,
    }


def _model_pool(platform: str) -> Any:
    if platform == "gpt":
        return account_service
    if platform == "gemini":
        return gemini_account_service
    raise HTTPException(status_code=400, detail="仅 GPT 和 Gemini 支持网页模型发现")

# ── OpenAI accounts ──


@router.get("/api/accounts")
async def list_accounts():
    return {"accounts": account_service.list_accounts()}


@router.delete("/api/accounts/{email}")
async def delete_account(email: str):
    success = account_service.delete_account(email)
    if not success:
        raise HTTPException(status_code=404, detail="Account not found")
    return {"code": 0, "message": f"Account {email} deleted"}


@router.post("/api/accounts/{email}/refresh")
async def refresh_account(email: str):
    try:
        acc = account_service.refresh_account(email)
        return {"code": 0, "message": f"Account {email} refreshed", "data": acc["email"]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/accounts/{email}/test")
async def test_openai_account(email: str):
    with account_service._lock:
        account = account_service._accounts.get(email)
        if account:
            account = dict(account)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    t0 = time.time()
    try:
        def _test():
            with OpenAIBackendAPI(access_token=account["access_token"], proxy=account.get("proxy", "")) as backend:
                backend.chat_text("Hi", model="gpt-5-5")
        await asyncio.to_thread(_test)
        account_service.release_account(email, success=True)
        return {"code": 0, "message": "Account is healthy", "elapsed": f"{time.time() - t0:.2f}s"}
    except Exception as exc:
        account_service.release_account(email, success=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Test failed: {exc}")


# ── Doubao accounts ──


@router.get("/api/doubao/accounts")
async def list_doubao_accounts():
    return {"accounts": doubao_account_service.list_accounts()}


@router.post("/api/doubao/accounts")
async def add_doubao_account(body: DoubaoAccountRequest):
    try:
        account = doubao_account_service.add_account(body.name, body.cookie, body.proxy)
        return {"name": account["name"], "status": account["status"]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api/doubao/accounts/{name}")
async def delete_doubao_account(name: str):
    if not doubao_account_service.delete_account(name):
        raise HTTPException(status_code=404, detail="Doubao account not found")
    return {"status": "deleted"}


@router.get("/api/doubao/stats")
async def doubao_stats():
    return doubao_account_service.stats()


@router.post("/api/doubao/accounts/{name}/test")
async def test_doubao_account(name: str):
    with doubao_account_service._lock:
        account = doubao_account_service._accounts.get(name)
        if account:
            account = dict(account)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    t0 = time.time()
    try:
        async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
            await backend.chat("Hi")
        doubao_account_service.release_account(name, success=True)
        return {"code": 0, "message": "Account is healthy", "elapsed": f"{time.time() - t0:.2f}s"}
    except Exception as exc:
        doubao_account_service.release_account(name, success=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Test failed: {exc}")


# ── Gemini accounts ──


@router.get("/api/gemini/accounts")
async def list_gemini_accounts():
    return {"accounts": gemini_account_service.list_accounts()}


@router.post("/api/gemini/accounts")
async def add_gemini_account(body: GeminiAccountRequest):
    try:
        await gemini_account_service.discard_client(body.name.strip())
        account = gemini_account_service.add_account(body.name, body.cookie, body.proxy)
        # 新录入 Cookie 立即建立账号级客户端；成功后自动续期任务才会常驻。
        await gemini_account_service.get_client(account)
        return {"name": account["name"], "status": account["status"]}
    except Exception as exc:
        if "account" in locals():
            gemini_account_service.mark_refresh_verification_failed(account["name"], exc)
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api/gemini/accounts/{name}")
async def delete_gemini_account(name: str):
    if not gemini_account_service.delete_account(name):
        raise HTTPException(status_code=404, detail="Gemini account not found")
    await gemini_account_service.discard_client(name)
    return {"status": "deleted"}


@router.get("/api/gemini/stats")
async def gemini_stats():
    return gemini_account_service.stats()


@router.post("/api/gemini/accounts/{name}/test")
async def test_gemini_account(name: str):
    with gemini_account_service._lock:
        account = gemini_account_service._accounts.get(name)
        if account:
            account = dict(account)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    t0 = time.time()
    try:
        async with GeminiBackendAPI(account) as backend:
            if not backend.client._check_account_status():
                raise RuntimeError("Gemini Cookie 未认证，请更新完整 Cookie Header")
            result = await backend.client.generate_content("请只回复 OK。", temporary=True)
            if not result.text.strip():
                raise RuntimeError("Gemini 测试请求未返回文本")
        gemini_account_service.release_account(name, success=True)
        return {"code": 0, "message": "Account is healthy", "elapsed": f"{time.time() - t0:.2f}s"}
    except Exception as exc:
        gemini_account_service.release_account(name, success=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Test failed: {exc}")


# ── Model configs CRUD ──


@router.get("/api/model-configs")
async def list_model_configs():
    return {"configs": database.list_model_configs()}


@router.get("/api/model-capabilities/{platform}")
async def list_model_capabilities(platform: str):
    return _model_capability_view(platform, _model_pool(platform))


@router.post("/api/model-capabilities/{platform}/refresh")
async def refresh_model_capabilities(platform: str):
    pool = _model_pool(platform)
    key = "email" if platform == "gpt" else "name"
    results = {}
    for account in pool.list_accounts():
        account_key = account.get(key, "")
        if not account_key or account.get("status") != "active":
            continue
        try:
            if platform == "gpt":
                models = await asyncio.to_thread(pool.refresh_supported_models, account_key)
            else:
                models = await pool.refresh_supported_models(account_key)
            results[account_key] = {"status": "ok", "count": len(models)}
        except Exception as exc:
            results[account_key] = {"status": "error", "error": str(exc)[:500]}

    return {"results": results, **_model_capability_view(platform, pool)}


@router.get("/api/model-configs/options")
async def get_model_options():
    """返回当前账号实际发现的模型，避免展示已下线的静态名称。"""
    # 1. Gemini：模型注册表是唯一来源，不能按名称猜测生图能力。
    gemini_dynamic = gemini_account_service.get_available_models()

    # 2. GPT (OpenAI)
    gpt_dynamic = account_service.get_available_models() if hasattr(account_service, "get_available_models") else []
    gpt_image_presets = [
        {"value": "gpt-image-2", "label": "gpt-image-2 (推荐生图)", "description": "ChatGPT 原生 4o-Canvas 生图通道"},
        {"value": "gpt-4o", "label": "gpt-4o", "description": "多模态视觉绘图支持"},
    ]
    gpt_chat_presets = [
        {"value": "gpt-5-5", "label": "gpt-5-5 (推荐对话)", "description": "ChatGPT 默认对话模型"},
        {"value": "gpt-4o", "label": "gpt-4o", "description": "全能旗舰多模态模型"},
        {"value": "gpt-4o-mini", "label": "gpt-4o-mini", "description": "轻量高速日常模型"},
        {"value": "o1", "label": "o1", "description": "深度思维链强化学习模型"},
        {"value": "o3-mini", "label": "o3-mini", "description": "最新高推理效率模型"},
    ]
    gpt_image_map = {item["value"]: item for item in gpt_image_presets}
    gpt_chat_map = {item["value"]: item for item in gpt_chat_presets}
    for item in gpt_dynamic:
        gpt_chat_map[item["value"]] = item
        if "image" in item["value"]:
            gpt_image_map[item["value"]] = item

    # 3. 豆包 (Doubao)
    doubao_options = [
        {"value": "default", "label": "default (系统默认)", "description": "豆包网页端全托管助手"}
    ]

    return {
        "gemini": {
            "image_models": gemini_dynamic,
            "chat_models": gemini_dynamic,
        },
        "gpt": {
            "image_models": list(gpt_image_map.values()),
            "chat_models": list(gpt_chat_map.values()),
        },
        "doubao": {
            "image_models": doubao_options,
            "chat_models": doubao_options,
        },
    }


@router.put("/api/model-configs/{platform}")
async def update_model_config(platform: str, body: ModelConfigUpdateRequest):
    if platform not in {"gpt", "gemini", "doubao"}:
        raise HTTPException(status_code=400, detail="不支持的平台类型，仅限 gpt, gemini, doubao")
    database.update_model_config(platform, body.image_model, body.chat_model, body.description)
    return {"code": 0, "message": f"{platform} 模型配置已更新", "data": {"platform": platform, "image_model": body.image_model, "chat_model": body.chat_model}}
