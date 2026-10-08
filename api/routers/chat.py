from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import ChatCompletionRequest
from core.router import resolve_model
from core.admission import CapacityUnavailable
from api.model_execution import acquire_traced_account, execute_model_request
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from providers.openai.credentials import CredentialUnavailable
from utils.helper import UpstreamHTTPError, is_transport_error
from utils.log import logger, proxy_log_ref, python_attempt_log_context, stable_log_ref

router = APIRouter(tags=["chat"])

MULTIMODAL_ANALYSIS_CONSTRAINT = (
    "\n\n【输出要求】：请直接输出纯文本分析报告与文字构图方案，无需生成图片。"
)


def apply_analysis_constraint(prompt: str) -> str:
    """确保分析类 Prompt 末尾包含显式纯文本约束，防止模型误判而直接去调用生图。"""
    trimmed = prompt.strip()
    if "无需生成图片" not in trimmed and "【输出要求】" not in trimmed:
        return trimmed + MULTIMODAL_ANALYSIS_CONSTRAINT
    return trimmed


def is_json_response(text: str) -> bool:
    """
    智能识别模型输出是否属于合法的 JSON 格式。
    支持纯 JSON 对象/数组以及 Markdown 代码块（```json ... ```）包裹的 JSON。
    用于对 Phase 6 产品图溯源回填等结构化返回进行无条件豁免，防止被字数校验误杀。
    """
    trimmed = text.strip()
    # 纯 JSON 结构直接判定
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
        try:
            json.loads(trimmed)
            return True
        except Exception:
            pass

    # Markdown 代码块包裹的 JSON
    if "```" in trimmed:
        match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", trimmed, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1).strip())
                return True
            except Exception:
                pass

    return False


def is_anomalous_text(text: str) -> bool:
    """
    判断模型返回文本是否命中异常场景：
    1. 若为合法的 JSON 格式，直接放行（豁免字数拦截）；
    2. 若包含【推荐产品图】等精简选品指令，直接放行（豁免 40 字下限拦截，防止 Phase 2B 误杀）；
    3. 字数 < 40：判定为触发安全风控截断、连接异常中断、或模型误把分析当成生图导致文本极短；
    4. 字数 > 4000：判定为模型出现严重冗余发散与幻觉。
    """
    if is_json_response(text):
        return False
    trimmed = text.strip()
    # 豁免【推荐产品图】选品指令，允许 1~39 字精简返回
    if "【推荐产品图】" in trimmed or "推荐产品图" in trimmed:
        return False
    text_len = len(trimmed)
    return text_len < 40 or text_len > 4000


async def _execute_chat_once(resolved: Any, prompt: str, images: list[str], request_id: str, attempt: int) -> str:
    """执行单次大模型多模态对话推理，管理对应平台的账号生命周期。"""
    if resolved.platform == "gemini":
        account = await acquire_traced_account(gemini_account_service, "chat", "gemini", "name", attempt)
        account_ref = stable_log_ref("gemini-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with GeminiBackendAPI(account) as backend:
                text = await backend.chat(prompt, images, resolved.model)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            gemini_account_service.release_account(account["name"], True, task_type="chat", acquired_account=account)
            return text
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_finished outcome=cancelled request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            gemini_account_service.release_account(account["name"], False, task_type="chat", acquired_account=account)
            raise
        except Exception as exc:
            logger.warning("event=upstream_attempt_finished outcome=failed request_id=%s python_attempt=%d platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d reason_code=%s http_status=%s failure_scope=%s",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000),
                           type(exc).__name__, getattr(exc, "status_code", None), "transport" if is_transport_error(exc) else "account")
            # 本次连接已由 GeminiBackendAPI 单独回收，其他并发请求继续执行。
            gemini_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="chat", failure_scope="transport" if is_transport_error(exc) else "account", acquired_account=account,
            )
            raise
    elif resolved.platform == "doubao":
        account = await acquire_traced_account(doubao_account_service, "chat", "doubao", "name", attempt)
        account_ref = stable_log_ref("doubao-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with DoubaoBackendAPI(account["cookies"], account.get("proxy", ""), account.get("device_id"), account.get("web_id"), account.get("fp")) as backend:
                attachments = [await backend.upload_image(url) for url in images]
                text = await backend.chat(prompt, attachments)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            doubao_account_service.release_account(account["name"], True, task_type="chat", acquired_account=account)
            return text
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_finished outcome=cancelled request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            doubao_account_service.release_account(account["name"], False, task_type="chat", acquired_account=account)
            raise
        except Exception as exc:
            logger.warning("event=upstream_attempt_finished outcome=failed request_id=%s python_attempt=%d platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d reason_code=%s http_status=%s failure_scope=%s",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000),
                           type(exc).__name__, getattr(exc, "status_code", None), "transport" if is_transport_error(exc) else "account")
            doubao_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="chat", failure_scope="transport" if is_transport_error(exc) else "account", acquired_account=account,
            )
            raise
    else:
        account = await acquire_traced_account(account_service, "chat", "gpt", "email", attempt)
        account_ref = stable_log_ref("gpt-account", account.get("email"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s",
                    request_id, attempt, account_ref, proxy_ref)
        upstream_started = time.monotonic()
        try:
            async with OpenAIBackendAPI(account["access_token"], account.get("proxy", ""), account.get("device_id", ""),
                    credential_provider=lambda: account_service.prepare_request_account(account)) as backend:
                text = await backend.chat_text(prompt, images, resolved.model)
            account_service.release_account(account["email"], True, task_type="chat", acquired_account=account)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            return text
        except asyncio.CancelledError:
            logger.warning("event=upstream_attempt_finished outcome=cancelled request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            account_service.release_account(account["email"], False, task_type="chat", acquired_account=account)
            raise
        except Exception as exc:
            logger.warning("event=upstream_attempt_finished outcome=failed request_id=%s python_attempt=%d platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d reason_code=%s http_status=%s failure_scope=%s",
                           request_id, attempt, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000),
                           type(exc).__name__, getattr(exc, "status_code", None), "transport" if is_transport_error(exc) else "account")
            account_service.release_account(
                account["email"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="chat", failure_scope="transport" if is_transport_error(exc) else "account", acquired_account=account,
            )
            raise



@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    resolved = resolve_model(body.model, "chat")
    request_id = body.request_id or uuid.uuid4().hex
    body = body.model_copy(update={"request_id": request_id})
    return await execute_model_request(resolved.platform, request_id, lambda: _chat_completions(body),
                                       request=request, java_attempt=body.java_attempt)


async def _chat_completions(body: ChatCompletionRequest):
    """
    多模态对话与分析接口：
    具备代理侧智能文本校验与就地自动重试机制。
    针对非 JSON 格式的返回，如果字数 < 40（风控截断/误生图）或 > 4000（失控发散），自动在代理内就地重试 1 次。
    """
    resolved = resolve_model(body.model, "chat")
    prompt = apply_analysis_constraint(body.prompt)
    request_id = body.request_id or uuid.uuid4().hex
    started_at = time.monotonic()
    logger.info(
        "event=chat_request_started request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s model=%s images=%d",
        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
        body.stage, resolved.platform, resolved.model, len(body.images),
    )

    # 包含首次调用与 1 次异常就地重试，最大尝试 2 次
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            # 账号池内部的状态事件复用当前尝试序号，便于区分就地重试。
            attempt_token = python_attempt_log_context.set(attempt + 1)
            try:
                text = await _execute_chat_once(resolved, prompt, body.images, request_id, attempt + 1)
            finally:
                python_attempt_log_context.reset(attempt_token)

            # 校验文本是否命中字数异常
            if is_anomalous_text(text):
                text_len = len(text.strip())
                anomaly_detail = (
                    f"模型返回文本字数异常 (当前 {text_len} 字，要求 40~4000 字，"
                    f"疑似风控拦截截断、要求出文字却误调用生图、或内容发散超长)"
                )
                if attempt < max_attempts - 1:
                    logger.warning(
                        "event=chat_attempt_anomalous request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s python_attempt=%d max_attempts=%d output_chars=%d",
                        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                        body.stage, attempt + 1, max_attempts, text_len,
                    )
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(
                        "event=chat_request_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s python_attempt=%d max_attempts=%d elapsed_ms=%d reason=anomalous_output output_chars=%d",
                        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                        body.stage, attempt + 1, max_attempts, int((time.monotonic() - started_at) * 1000), text_len,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"{resolved.platform} chat output anomaly: {anomaly_detail}",
                        headers={"X-Request-ID": request_id},
                    )

            # 正常合规文本或合法 JSON，直接成功返回
            logger.info(
                "event=chat_request_succeeded request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s python_attempt=%d elapsed_ms=%d output_chars=%d",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, attempt + 1, int((time.monotonic() - started_at) * 1000), len(text),
            )
            return JSONResponse(
                content={"code": 0, "model": body.model, "text": text, "created": int(time.time())},
                headers={"X-Request-ID": request_id},
            )
        except CapacityUnavailable:
            raise
        except CredentialUnavailable as exc:
            raise HTTPException(status_code=502, detail="GPT credential preparation failed",
                                headers={"X-Request-ID": request_id}) from exc
        except HTTPException:
            raise
        except Exception as exc:
            if attempt < max_attempts - 1:
                logger.warning(
                    "event=chat_attempt_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s python_attempt=%d max_attempts=%d reason=%s",
                    request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                    body.stage, resolved.platform, attempt + 1, max_attempts, type(exc).__name__,
                )
                await asyncio.sleep(1)
                continue
            logger.error(
                "event=chat_request_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s python_attempt=%d max_attempts=%d elapsed_ms=%d reason=%s",
                request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
                body.stage, resolved.platform, attempt + 1, max_attempts,
                int((time.monotonic() - started_at) * 1000), type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail=f"{resolved.platform} chat failed: {exc}",
                                headers={"X-Request-ID": request_id}) from exc
