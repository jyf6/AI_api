from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import DoubaoAccountRequest, GeminiAccountRequest
from providers.openai.account import account_service
from providers.doubao.account import doubao_account_service
from providers.gemini.account import gemini_account_service
from providers.openai.backend import OpenAIBackendAPI
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.backend import GeminiBackendAPI
import time
import asyncio

router = APIRouter(tags=["accounts"])

_HEALTH_CHECK_PROMPT = "请只回复 OK。"


def _validate_health_check_response(response: str) -> None:
    normalized = str(response or "").strip().strip("`*_\"'“”‘’。.!！ \t\r\n").lower()
    if normalized != "ok":
        raise RuntimeError("Model connectivity test did not return the expected OK response")

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
                return backend.chat_text(_HEALTH_CHECK_PROMPT, model="auto")
        response = await asyncio.to_thread(_test)
        _validate_health_check_response(response)
        account_service.set_account_health(email, healthy=True)
        return {"code": 0, "message": "Account is healthy", "elapsed": f"{time.time() - t0:.2f}s"}
    except Exception as exc:
        if account_service._classify_error(str(exc), getattr(exc, "status_code", None)) == "fatal":
            account_service.set_account_health(email, healthy=False, error=str(exc))
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
            response = await backend.chat(_HEALTH_CHECK_PROMPT)
        _validate_health_check_response(response)
        doubao_account_service.set_account_health(name, healthy=True)
        return {"code": 0, "message": "Account is healthy", "elapsed": f"{time.time() - t0:.2f}s"}
    except Exception as exc:
        if doubao_account_service._classify_error(str(exc), getattr(exc, "status_code", None)) == "fatal":
            doubao_account_service.set_account_health(name, healthy=False, error=str(exc))
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
            result = await backend.client.generate_content(_HEALTH_CHECK_PROMPT, temporary=True)
            _validate_health_check_response(result.text)
        gemini_account_service.set_account_health(name, healthy=True)
        return {"code": 0, "message": "Gemini text connectivity test passed", "elapsed": f"{time.time() - t0:.2f}s"}
    except Exception as exc:
        if gemini_account_service.is_auth_error(exc):
            await gemini_account_service.discard_client(name)
            gemini_account_service.set_account_health(name, healthy=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Test failed: {exc}")
