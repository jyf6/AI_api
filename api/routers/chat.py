from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import ChatCompletionRequest
from core.router import resolve_model
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from utils.helper import UpstreamHTTPError
from utils.log import logger

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


async def _execute_chat_once(resolved: Any, prompt: str, images: list[str]) -> str:
    """执行单次大模型多模态对话推理，管理对应平台的账号生命周期。"""
    if resolved.platform == "gemini":
        account = await asyncio.to_thread(gemini_account_service.wait_for_available_account, "chat")
        try:
            async with GeminiBackendAPI(account) as backend:
                text = await backend.chat(prompt, images, resolved.model)
            gemini_account_service.release_account(account["name"], True, task_type="chat")
            return text
        except asyncio.CancelledError:
            gemini_account_service.release_account(account["name"], False, "request cancelled", task_type="chat")
            raise
        except Exception as exc:
            # 本次连接已由 GeminiBackendAPI 单独回收，其他并发请求继续执行。
            gemini_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="chat",
            )
            raise
    elif resolved.platform == "doubao":
        account = await asyncio.to_thread(doubao_account_service.wait_for_available_account, "chat")
        try:
            async with DoubaoBackendAPI(account["cookies"], account.get("proxy", "")) as backend:
                attachments = [await backend.upload_image(url) for url in images]
                text = await backend.chat(prompt, attachments)
            doubao_account_service.release_account(account["name"], True, task_type="chat")
            return text
        except asyncio.CancelledError:
            doubao_account_service.release_account(account["name"], False, "request cancelled", task_type="chat")
            raise
        except Exception as exc:
            doubao_account_service.release_account(
                account["name"], False, str(exc), status_code=getattr(exc, "status_code", None),
                task_type="chat",
            )
            raise
    else:
        account = await asyncio.to_thread(account_service.wait_for_available_account, "chat")
        try:
            def run() -> str:
                with OpenAIBackendAPI(account["access_token"], account.get("proxy", "")) as backend:
                    return backend.chat_text(prompt, images, resolved.model)
            text = await asyncio.to_thread(run)
            account_service.release_account(account["email"], True, task_type="chat")
            return text
        except asyncio.CancelledError:
            account_service.release_account(account["email"], False, "request cancelled", task_type="chat")
            raise
        except Exception as exc:
            status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
            account_service.release_account(account["email"], False, str(exc), status_code=status, task_type="chat")
            raise


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    """
    多模态对话与分析接口：
    具备代理侧智能文本校验与就地自动重试机制。
    针对非 JSON 格式的返回，如果字数 < 40（风控截断/误生图）或 > 4000（失控发散），自动在代理内就地重试 1 次。
    """
    resolved = resolve_model(body.model, "chat")
    prompt = apply_analysis_constraint(body.prompt)

    # 包含首次调用与 1 次异常就地重试，最大尝试 2 次
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            text = await _execute_chat_once(resolved, prompt, body.images)

            # 校验文本是否命中字数异常
            if is_anomalous_text(text):
                text_len = len(text.strip())
                anomaly_detail = (
                    f"模型返回文本字数异常 (当前 {text_len} 字，要求 40~4000 字，"
                    f"疑似风控拦截截断、要求出文字却误调用生图、或内容发散超长)"
                )
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"[Chat Anomaly] {anomaly_detail}，触发代理侧自动就地重试 (第 {attempt + 1}/{max_attempts - 1} 次)... "
                        f"原输出片段: {text.strip()[:60]!r}"
                    )
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"[Chat Anomaly] {anomaly_detail}，已达到最大重试次数，向客户端返回错误")
                    raise HTTPException(
                        status_code=502,
                        detail=f"{resolved.platform} chat output anomaly: {anomaly_detail}"
                    )

            # 正常合规文本或合法 JSON，直接成功返回
            return {"code": 0, "model": body.model, "text": text, "created": int(time.time())}
        except HTTPException:
            raise
        except Exception as exc:
            if attempt < max_attempts - 1:
                logger.warning(
                    f"[Chat Retry] 调用 {resolved.platform} 发生网络/接口异常: {exc}，将在 1 秒后自动进行第 {attempt + 1} 次就地重试..."
                )
                await asyncio.sleep(1)
                continue
            raise HTTPException(status_code=502, detail=f"{resolved.platform} chat failed: {exc}") from exc
