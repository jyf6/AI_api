from __future__ import annotations

import logging
import hashlib
import time
from contextlib import contextmanager
from contextvars import ContextVar


# asyncio 请求内的日志上下文；不同并发任务互不覆盖，后台任务保持空值。
request_log_context: ContextVar[tuple[str, int | None] | None] = ContextVar("request_log_context", default=None)
python_attempt_log_context: ContextVar[int | None] = ContextVar("python_attempt_log_context", default=None)


class RequestLogFilter(logging.Filter):
    """为账号池等深层日志补上调用来源，不要求逐层修改业务函数签名。"""

    def filter(self, record: logging.LogRecord) -> bool:
        context = request_log_context.get()
        if context is not None and not getattr(record, "_request_context_added", False):
            message = record.getMessage()
            request_id, java_attempt = context
            if "request_id=" not in message:
                message += f" request_id={request_id}"
            python_attempt = python_attempt_log_context.get()
            if python_attempt is not None and "python_attempt=" not in message:
                message += f" python_attempt={python_attempt}"
            record.msg = f"{message} java_attempt={java_attempt}"
            record.args = ()
            record._request_context_added = True
        return True

logger = logging.getLogger("chatgpt-image-service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.addFilter(RequestLogFilter())
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03dZ] [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def stable_log_ref(namespace: str, value: str | None) -> str:
    """为账号或代理生成稳定摘要，供跨请求排障时关联且不直接暴露原值。"""
    if not value:
        return "direct" if namespace == "proxy" else "unknown"
    raw = f"{namespace}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def proxy_log_ref(account: dict) -> str:
    """账号已有节点 ID 时以 ID 关联日志，节点换地址后历史引用仍稳定。"""
    proxy_id = account.get("proxy_id")
    if not proxy_id and not account.get("proxy"):
        return "direct"
    return stable_log_ref("proxy-node", str(proxy_id) if proxy_id else account.get("proxy"))


def error_http_status(error: BaseException) -> int | None:
    return getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)


@contextmanager
def image_stage(platform: str, stage: str):
    """Record the boundary and duration of a generated-image stage."""
    started = time.monotonic()
    logger.info("event=image_stage_started platform=%s stage=%s", platform, stage)
    try:
        yield
    except BaseException as exc:
        logger.warning(
            "event=image_stage_finished platform=%s stage=%s outcome=%s duration_ms=%d reason_code=%s http_status=%s transport_code=%s",
            platform, stage, "cancelled" if type(exc).__name__ == "CancelledError" else "failed",
            int((time.monotonic() - started) * 1000), type(exc).__name__, error_http_status(exc),
            getattr(exc, "code", None),
        )
        raise
    else:
        logger.info("event=image_stage_finished platform=%s stage=%s outcome=success duration_ms=%d",
                    platform, stage, int((time.monotonic() - started) * 1000))
