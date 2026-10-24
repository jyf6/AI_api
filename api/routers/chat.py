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
from core.model_timeout import MODEL_TIMEOUT_SECONDS
from core.admission import CapacityUnavailable
from api.model_execution import acquire_traced_account, execute_model_request
from providers.doubao.account import doubao_account_service
from providers.doubao.backend import DoubaoBackendAPI
from providers.gemini.account import gemini_account_service
from providers.gemini.backend import GeminiBackendAPI
from providers.gemini.webapi.exceptions import AuthError, TemporarilyBlockedError, UsageLimitExceededError
from providers.openai.account import account_service
from providers.openai.backend import OpenAIBackendAPI
from providers.openai.credentials import CredentialUnavailable
from utils.helper import (QuotaExhaustedError, UpstreamHTTPError, detect_quota_exhausted,
                          is_transport_error)
from utils.log import error_http_status, logger, proxy_log_ref, python_attempt_log_context, stable_log_ref

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


def get_anomaly_correction_prompt(text: str, stage: str, image_count: int | None = None) -> tuple[bool, str | None]:
    """
    校验大模型多模态/文本分析输出是否异常，并在异常时给出纠偏追问 Prompt。
    规则执行顺序：
    1. 如果是产品选图阶段 (INITIAL_ITEM_PRODUCT_SELECTION)：
       - 严格置顶验证单行输出，禁止跨行；
       - 绝不被任何 JSON 规则或推荐产品图豁免提前放行；
       - 格式必须严格匹配：'Selected product images: None' 或 'Selected product images: 1, 2'；
       - 若包含编号，必须校验编号合法性（>= 1，若已知素材拼图数量，不能超出拼图上限）；
       - 不合规时返回格式模板追问；
    2. 其他分析阶段：
       - 合法 JSON 格式直接放行；
       - 字数少于 40 个字时返回 '请重试'（自愈软拒答）；
       - 参考图分析与构图分析不设字数上限，其他全局分析阶段若 > 4000 字拦截；
       - 其余 >= 40 字直接放行。
    """
    trimmed = text.strip()

    # 1. 产品选图阶段：严格置顶校验
    if stage == "INITIAL_ITEM_PRODUCT_SELECTION":
        # 禁止跨行输出
        if "\n" in trimmed or "\r" in trimmed:
            return True, "格式不对。必须仅输出单行内容，不得跨行或包含任何多余文字：\nSelected product images: 1, 2\n（若无需产品则输出 Selected product images: None）"

        # 检查显式合法空选择: 必须是标准大小写 'None'
        if trimmed == "Selected product images: None":
            return False, None

        # 严格单行编号匹配 (注意: selected product images: none 等大小写不符合视为不通过)
        match = re.fullmatch(r"Selected product images:[ \t]*([0-9]+(?:[ \t]*[,，][ \t]*[0-9]+)*)", trimmed)
        if not match:
            return True, "格式不对。请严格按照标准格式重新输出，不要包含任何多余文字或解释：\nSelected product images: 1, 2\n（若不需要产品图则输出 Selected product images: None）"

        # 验证提取出的每一个编号
        numbers_str = match.group(1)
        numbers = [int(num.strip()) for num in re.split(r"[,，\s]+", numbers_str) if num.strip()]
        if not numbers:
            return True, "格式不对。必须指定有效的产品图编号：\nSelected product images: 1, 2\n（若不需要产品图则输出 Selected product images: None）"

        if any(num < 1 for num in numbers):
            return True, "产品图编号必须大于等于 1。请重新核对拼图上的编号并按标准格式输出：\nSelected product images: 1, 2"

        # 若提供了素材拼图数，每个拼图为九宫格（最多 9 张），校验是否超出上限
        if image_count is not None and image_count > 0:
            max_allowed = image_count * 9
            if any(num > max_allowed for num in numbers):
                return True, f"产品图编号超出当前素材范围（当前共 {image_count} 张素材拼图，最大编号为 {max_allowed}）。请核对拼图角标后重新输出：\nSelected product images: 1, 2"

        return False, None

    # 2. 其他分析阶段：JSON 格式豁免
    if is_json_response(text):
        return False, None

    # 3. 字数校验与长文本控制（删除“包含推荐产品图就放行”的非格式豁免，普通分析严格执行字数与质量判定）
    text_len = len(trimmed)
    if stage in {"INITIAL_ITEM_REF_ANALYSIS", "INITIAL_ITEM_COMPOSITION_PLANNING"}:
        if text_len < 40:
            return True, "请重试"
        return False, None

    if text_len < 40:
        return True, "请重试"

    return False, None


def is_anomalous_text(text: str, stage: str, image_count: int | None = None) -> bool:
    anomaly, _ = get_anomaly_correction_prompt(text, stage, image_count)
    return anomaly


def analysis_failure_detail(platform: str, stage: str, request_id: str, reason: str) -> str:
    """用现有文本错误协议保留阶段和真实原因，脱敏后可直接落库及展示。"""
    stages = {
        "INITIAL_TASK_INSTRUCTION_ANALYSIS": "用户指令分析",
        "INITIAL_GLOBAL_ANALYSIS": "全局风格分析",
        "INITIAL_ITEM_REF_ANALYSIS": "参考图分析",
        "INITIAL_ITEM_COMPOSITION_PLANNING": "构图规划",
        "INITIAL_ITEM_PRODUCT_SELECTION": "产品图选择",
    }
    reason = re.sub(r"(?i)(Bearer\s+)[^\s\"',;]+", r"\1***", reason)
    reason = re.sub(
        r"(?i)((?:access_token|refresh_token|token|api_key|password|cookie|__Secure-1PSID(?:TS)?)[\"']?\s*[:=]\s*[\"']?)[^\"'\s,;]+",
        r"\1***", reason,
    )
    return f"{platform.upper()} 文本分析失败（阶段：{stages.get(stage, stage or '文本分析')}）：{reason}；请求编号：{request_id}"


def invalid_analysis_detail(platform: str, stage: str, request_id: str, text: str, correction: str) -> str:
    """同会话纠偏仍失败时说明具体校验条件，避免笼统归因为生图失败。"""
    reason = (f"返回内容过短：{len(text.strip())} 个字符，要求至少 40 个字符或合法 JSON"
              if correction == "请重试" else correction)
    return analysis_failure_detail(platform, stage, request_id, f"输出校验失败，纠偏重试后仍不合规：{reason}")


def _check_quota_text(text: str, service, account_key: str, task_type: str, acquired_account: dict,
                      platform: str, request_id: str) -> None:
    """返回不到 40 字的文本先做额度检查（JSON 由检测函数豁免），命中即冷却账号，不重试。"""
    stripped = (text or "").strip()
    if len(stripped) >= 40:
        return
    is_quota, wait_sec = detect_quota_exhausted(stripped)
    if not is_quota:
        return
    # 记录失败让账号池进入 quota_exhausted 冷却，本次请求不重试。
    service.release_account(account_key, False, stripped, task_type=task_type,
                            failure_scope="account", acquired_account=acquired_account)
    wait_min = max(1, wait_sec // 60)
    raise HTTPException(
        status_code=429,
        detail=json.dumps({
            "code": "ACCOUNT_QUOTA_EXHAUSTED",
            "reason": "ACCOUNT_QUOTA_EXHAUSTED",
            "platform": platform,
            "message": f"当前账号额度已耗尽，预计将在 {wait_min} 分钟后恢复。可等待恢复或确认切换账号。",
            "userActions": ["WAIT", "SWITCH_ACCOUNT"],
            "upstreamState": "NOT_SENT",
        }, ensure_ascii=False),
        headers={"X-Request-ID": request_id, "Retry-After": str(wait_sec)},
    )


async def _execute_chat_with_in_session_retry(
    resolved: Any, prompt: str, images: list[str], request_id: str, stage: str, thinking_level: str = "medium"
) -> str:
    """
    执行大模型多模态对话推理，管理对应平台的账号生命周期。
    在同一个账号、同一个会话内进行异常自愈重试：
    首轮若返回异常（字数 < 40 或选图格式不符），保持同一个账号和会话实例，等待 3 秒后下发追问（'请重试' 或标准选图格式）。
    2 次均不合规时返回具体的输出校验失败原因。
    """
    upstream_started = time.monotonic()
    if resolved.platform == "gemini":
        # Gemini 分析模型固定使用最低思考，不受请求档位影响。
        thinking_level = "low"
        account = await acquire_traced_account(gemini_account_service, "chat", "gemini", "name", 1)
        account_ref = stable_log_ref("gemini-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s platform=gemini account_ref=%s proxy_ref=%s stage=%s",
                    request_id, account_ref, proxy_ref, stage)
        try:
            async with GeminiBackendAPI(account,
                    before_model_send=lambda: gemini_account_service.wait_for_model_send(account),
                    after_model_send=lambda: gemini_account_service.record_model_call(account)) as backend:
                # 第一轮推理（保持 chat 会话对象）
                text, chat = await backend.chat_with_session(prompt, images, resolved.model, thinking_level)
                _check_quota_text(text, gemini_account_service, account["name"], "chat", account,
                                  "gemini", request_id)
                is_anomaly, correction_prompt = get_anomaly_correction_prompt(text, stage, image_count=len(images))
                if is_anomaly:
                    logger.warning("event=in_session_anomaly_detected platform=gemini account_ref=%s stage=%s text_len=%d correction=%s",
                                   account_ref, stage, len(text.strip()), correction_prompt)
                    # 同账号同一个会话原地等待 3 秒后纠偏重试
                    await asyncio.sleep(3)
                    text2 = await backend.chat_continue(chat, correction_prompt, resolved.model, thinking_level)
                    _check_quota_text(text2, gemini_account_service, account["name"], "chat", account,
                                      "gemini", request_id)
                    is_anomaly2, correction2 = get_anomaly_correction_prompt(text2, stage, image_count=len(images))
                    if is_anomaly2:
                        logger.error("event=in_session_retry_failed platform=gemini account_ref=%s stage=%s text_len=%d",
                                     account_ref, stage, len(text2.strip()))
                        gemini_account_service.release_account(account["name"], True, task_type="chat", acquired_account=account)
                        raise HTTPException(status_code=502,
                                            detail=invalid_analysis_detail(resolved.platform, stage, request_id, text2, correction2),
                                            headers={"X-Request-ID": request_id})
                    text = text2

            gemini_account_service.release_account(account["name"], True, task_type="chat", acquired_account=account)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s platform=gemini account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            return text
        except HTTPException:
            raise
        except asyncio.CancelledError:
            gemini_account_service.release_account(account["name"], False, task_type="chat", acquired_account=account)
            raise
        except Exception as exc:
            from utils.helper import is_read_timeout_error
            is_timeout = is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            gemini_account_service.release_account(
                account["name"], False, str(exc),
                status_code=504 if is_timeout else (
                    401 if isinstance(exc, AuthError) else 429
                    if isinstance(exc, (TemporarilyBlockedError, UsageLimitExceededError))
                    else error_http_status(exc)
                ),
                task_type="chat", failure_scope="timeout" if is_timeout else ("transport" if is_transport_error(exc) else "account"),
                acquired_account=account,
            )
            raise

    elif resolved.platform == "doubao":
        account = await acquire_traced_account(doubao_account_service, "chat", "doubao", "name", 1)
        account_ref = stable_log_ref("doubao-account", account.get("name"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s platform=doubao account_ref=%s proxy_ref=%s stage=%s",
                    request_id, account_ref, proxy_ref, stage)
        try:
            async with DoubaoBackendAPI(account["cookies"], account.get("proxy", ""), account.get("device_id"), account.get("web_id"), account.get("fp")) as backend:
                attachments = [await backend.upload_image(url) for url in images]
                conv_id = str(uuid.uuid4())
                text, _ = await backend.chat_with_context(prompt, attachments, local_conversation_id=conv_id)
                _check_quota_text(text, doubao_account_service, account["name"], "chat", account,
                                  "doubao", request_id)
                is_anomaly, correction_prompt = get_anomaly_correction_prompt(text, stage, image_count=len(images))
                if is_anomaly:
                    logger.warning("event=in_session_anomaly_detected platform=doubao account_ref=%s stage=%s text_len=%d correction=%s",
                                   account_ref, stage, len(text.strip()), correction_prompt)
                    await asyncio.sleep(3)
                    text2, _ = await backend.chat_with_context(correction_prompt, [], local_conversation_id=conv_id, need_create_conversation=False)
                    _check_quota_text(text2, doubao_account_service, account["name"], "chat", account,
                                      "doubao", request_id)
                    is_anomaly2, correction2 = get_anomaly_correction_prompt(text2, stage, image_count=len(images))
                    if is_anomaly2:
                        logger.error("event=in_session_retry_failed platform=doubao account_ref=%s stage=%s text_len=%d",
                                     account_ref, stage, len(text2.strip()))
                        doubao_account_service.release_account(account["name"], True, task_type="chat", acquired_account=account)
                        raise HTTPException(status_code=502,
                                            detail=invalid_analysis_detail(resolved.platform, stage, request_id, text2, correction2),
                                            headers={"X-Request-ID": request_id})
                    text = text2

            doubao_account_service.release_account(account["name"], True, task_type="chat", acquired_account=account)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s platform=doubao account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            return text
        except HTTPException:
            raise
        except asyncio.CancelledError:
            doubao_account_service.release_account(account["name"], False, task_type="chat", acquired_account=account)
            raise
        except Exception as exc:
            from utils.helper import is_read_timeout_error
            is_timeout = is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            doubao_account_service.release_account(
                account["name"], False, str(exc),
                status_code=504 if is_timeout else (getattr(exc, "status_code", None) or error_http_status(exc)),
                task_type="chat", failure_scope="timeout" if is_timeout else ("transport" if is_transport_error(exc) else "account"),
                acquired_account=account,
            )
            raise

    else:
        account = await acquire_traced_account(account_service, "chat", "gpt", "email", 1)
        account_ref = stable_log_ref("gpt-account", account.get("email"))
        proxy_ref = proxy_log_ref(account)
        logger.info("event=upstream_attempt_started request_id=%s platform=gpt account_ref=%s proxy_ref=%s stage=%s",
                    request_id, account_ref, proxy_ref, stage)
        try:
            async with OpenAIBackendAPI(account["access_token"], account.get("proxy", ""), account.get("device_id", ""),
                    credential_provider=lambda: account_service.prepare_request_account(account),
                    before_model_send=lambda: account_service.wait_for_model_send(account),
                    after_model_send=lambda: account_service.record_model_call(account)) as backend:
                text, conv_id, msg_id = await backend.chat_text_with_context(
                    prompt, images, resolved.model, thinking_level=thinking_level
                )
                _check_quota_text(text, account_service, account["email"], "chat", account,
                                  "gpt", request_id)
                is_anomaly, correction_prompt = get_anomaly_correction_prompt(text, stage, image_count=len(images))
                if is_anomaly:
                    logger.warning("event=in_session_anomaly_detected platform=gpt account_ref=%s stage=%s text_len=%d correction=%s conv_id=%s",
                                   account_ref, stage, len(text.strip()), correction_prompt, conv_id)
                    await asyncio.sleep(3)
                    text2, _, _ = await backend.chat_text_with_context(
                        correction_prompt, None, resolved.model,
                        conversation_id=conv_id, parent_message_id=msg_id, thinking_level=thinking_level
                    )
                    _check_quota_text(text2, account_service, account["email"], "chat", account,
                                      "gpt", request_id)
                    is_anomaly2, correction2 = get_anomaly_correction_prompt(text2, stage, image_count=len(images))
                    if is_anomaly2:
                        logger.error("event=in_session_retry_failed platform=gpt account_ref=%s stage=%s text_len=%d",
                                     account_ref, stage, len(text2.strip()))
                        account_service.release_account(account["email"], True, task_type="chat", acquired_account=account)
                        raise HTTPException(status_code=502,
                                            detail=invalid_analysis_detail(resolved.platform, stage, request_id, text2, correction2),
                                            headers={"X-Request-ID": request_id})
                    text = text2

            account_service.release_account(account["email"], True, task_type="chat", acquired_account=account)
            logger.info("event=upstream_attempt_finished outcome=success request_id=%s platform=gpt account_ref=%s proxy_ref=%s duration_ms=%d",
                        request_id, account_ref, proxy_ref, int((time.monotonic() - upstream_started) * 1000))
            return text
        except HTTPException:
            raise
        except asyncio.CancelledError:
            account_service.release_account(account["email"], False, task_type="chat", acquired_account=account)
            raise
        except Exception as exc:
            from utils.helper import is_read_timeout_error
            is_timeout = is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            account_service.release_account(
                account["email"], False, str(exc),
                status_code=504 if is_timeout else (getattr(exc, "status_code", None) or error_http_status(exc)),
                retry_after=getattr(exc, "retry_after", None), task_type="chat",
                failure_scope="timeout" if is_timeout else ("transport" if is_transport_error(exc) else "account"),
                acquired_account=account,
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
    """多模态对话入口：初版多模态分析直接使用原始图片列表，无需在代理侧拼图。"""
    from utils.oss_reference import release_memory_references
    memory_keys = []
    try:
        return await _chat_completions_prepared(body)
    finally:
        release_memory_references(memory_keys)


async def _chat_completions_prepared(body: ChatCompletionRequest):
    """
    多模态对话与分析接口：
    具备代理侧智能文本校验与同一个账号同一个会话内的自动纠偏重试机制。
    针对非 JSON 格式的返回按阶段校验长度与格式，异常时在同一会话内等待 3 秒后重试 1 次。
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

    try:
        # 单次请求在同一账号、同一会话内完成首次推理与必要的自愈重试
        text = await asyncio.wait_for(
            _execute_chat_with_in_session_retry(resolved, prompt, body.images, request_id, body.stage, body.thinking_level),
            timeout=MODEL_TIMEOUT_SECONDS * 2,
        )
    except CapacityUnavailable:
        raise
    except CredentialUnavailable as exc:
        raise HTTPException(status_code=502,
                            detail=analysis_failure_detail(resolved.platform, body.stage, request_id,
                                                           f"CredentialUnavailable: {exc}"),
                            headers={"X-Request-ID": request_id}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "event=chat_request_failed request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s platform=%s elapsed_ms=%d reason=%s",
            request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
            body.stage, resolved.platform, int((time.monotonic() - started_at) * 1000), type(exc).__name__,
            exc_info=True,
        )

        from utils.helper import ImageQuotaExceededError, UpstreamHTTPError, is_read_timeout_error
        import json

        err_str = str(exc).lower()
        is_quota = isinstance(exc, (ImageQuotaExceededError, QuotaExhaustedError, UsageLimitExceededError)) or (
            detect_quota_exhausted(str(exc))[0]
        ) or any(
            kw in err_str for kw in ("quota", "upload_protection", "上限", "额度")
        )
        is_rate = isinstance(exc, TemporarilyBlockedError) or (
            isinstance(exc, UpstreamHTTPError) and exc.status_code == 429
        )

        if is_quota or is_rate:
            retry_after = getattr(exc, "retry_after", None)
            if is_quota:
                # 优先异常自带时间，其次从文本提取恢复时间，都没有则默认 5 小时。
                _, detected_seconds = detect_quota_exhausted(str(exc))
                wait_sec = (retry_after if retry_after and retry_after > 0
                            else detected_seconds if detected_seconds and detected_seconds > 0
                            else 18000)
                wait_min = max(1, wait_sec // 60)
                detail = {
                    "code": "ACCOUNT_QUOTA_EXHAUSTED",
                    "retry_after": wait_sec,
                    "retryAfter": wait_sec,
                    "cooldownUntil": int(time.time()) + wait_sec,
                    "message": f"当前账号额度已耗尽，预计将在 {wait_min} 分钟后恢复。如果切换账号会丢失历史记录，请选择是否切换账号"
                }
            else:
                wait_sec = 120
                detail = {
                    "code": "ACCOUNT_RATE_LIMITED",
                    "retry_after": wait_sec,
                    "retryAfter": wait_sec,
                    "cooldownUntil": int(time.time()) + wait_sec,
                    "message": "当前账号限流，请 2 分钟后稍后重试"
                }
            raise HTTPException(
                status_code=429,
                detail=json.dumps({**detail, "platform": resolved.platform}, ensure_ascii=False),
                headers={"X-Request-ID": request_id, "Retry-After": str(wait_sec)}
            )

        # 思考超时/读取超时统一对外收敛为 HTTP 504，提示保留原会话重试
        if is_read_timeout_error(exc) or isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or getattr(exc, "status_code", None) == 504:
            detail = {
                "code": "MODEL_READ_TIMEOUT",
                "platform": resolved.platform,
                "stage": body.stage or "text_analysis",
                "message": "模型文本分析读取超时，会话已保留；本次请求不会自动重新提交，请重试该分析步骤",
                "upstreamState": "SUBMITTED",
                "userActions": ["RETRY_CHAT"],
            }
            raise HTTPException(
                status_code=504,
                detail=json.dumps(detail, ensure_ascii=False),
                headers={"X-Request-ID": request_id},
            ) from exc

        # 保留异常链中的原始原因；沿用文本 detail，Java 和结果页可直接展示。
        reason = f"{type(exc).__name__}: {exc}"
        if exc.__cause__ is not None and str(exc.__cause__) not in reason:
            reason += f"；原始原因：{type(exc.__cause__).__name__}: {exc.__cause__}"
        raise HTTPException(status_code=502,
                            detail=analysis_failure_detail(resolved.platform, body.stage, request_id, reason),
                            headers={"X-Request-ID": request_id}) from exc

    # 正常合规文本或合法 JSON，直接成功返回
    logger.info(
        "event=chat_request_succeeded request_id=%s dispatch_id=%s task_code=%s operation_id=%s item_id=%s stage=%s elapsed_ms=%d output_chars=%d",
        request_id, body.dispatch_id, body.task_code, body.operation_id, body.item_id,
        body.stage, int((time.monotonic() - started_at) * 1000), len(text),
    )
    return JSONResponse(
        content={"code": 0, "model": body.model, "text": text, "created": int(time.time())},
        headers={"X-Request-ID": request_id},
    )
