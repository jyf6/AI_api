from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Header, HTTPException

from api.schemas import ChatCompletionRequest
from core.router import resolve_model
from core.database import database
from core.operations import begin_operation, replay_text
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from utils.helper import UpstreamHTTPError

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, operation_id: str | None = Header(default=None, alias="X-Operation-Id")):
    resolved = resolve_model(body.model, "chat")
    operation_id, existing = begin_operation(operation_id, "chat")
    if existing is not None:
        return replay_text(existing, operation_id)
    try:
        if resolved.platform == "gemini":
            account = await asyncio.to_thread(gemini_account_service.wait_for_available_account, "chat")
            try:
                async with GeminiBackendAPI(account) as backend:
                    text = await backend.chat(body.prompt, body.images, resolved.model)
                gemini_account_service.release_account(account["name"], True, task_type="chat")
            except Exception as exc:
                # 未得到文本即超时或失败时丢弃旧客户端，下一次随机调度会重新建立该账号连接。
                await gemini_account_service.discard_client(account["name"])
                gemini_account_service.release_account(account["name"], False, str(exc), task_type="chat")
                raise
        elif resolved.platform == "doubao":
            account = await asyncio.to_thread(doubao_account_service.wait_for_available_account, "chat")
            try:
                async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
                    attachments = [await backend.upload_image(url) for url in body.images]
                    text = await backend.chat(body.prompt, attachments)
                doubao_account_service.release_account(account["name"], True, task_type="chat")
            except Exception as exc:
                doubao_account_service.release_account(account["name"], False, str(exc), task_type="chat")
                raise
        else:
            account = await asyncio.to_thread(account_service.wait_for_available_account, "chat")
            try:
                def run() -> str:
                    with OpenAIBackendAPI(account["access_token"], account.get("proxy", "")) as backend:
                        return backend.chat_text(body.prompt, body.images, resolved.model)
                text = await asyncio.to_thread(run)
                account_service.release_account(account["email"], True, task_type="chat")
            except asyncio.CancelledError:
                account_service.release_account(account["email"], False, "request cancelled", task_type="chat")
                raise
            except Exception as exc:
                status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
                account_service.release_account(account["email"], False, str(exc), status_code=status, task_type="chat")
                raise
    except HTTPException:
        raise
    except Exception as exc:
        database.fail_operation(operation_id, str(exc))
        raise HTTPException(status_code=502, detail=f"{resolved.platform} chat failed: {exc}") from exc
    database.complete_operation(operation_id, text=text)
    return {"code": 0, "model": body.model, "text": text, "created": int(time.time()), "operation_id": operation_id}
