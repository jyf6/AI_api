from __future__ import annotations
import asyncio
import json
import re
from typing import Any, Iterator, AsyncIterator
from curl_cffi import requests

_UPSTREAM_BODY_LOG_LIMIT = 500

_CONNECT_EXCEPTION_NAMES = {
    "connectionerror", "connecterror", "connecttimeout", "dnserror",
    "clientconnectionerror", "clientconnectorerror", "clientconnectordnserror",
    "clientconnectorcertificateerror", "clienthttpproxyerror", "clientoserror",
    "clientproxyconnectionerror", "proxyerror", "proxyconnectionerror", "proxyconnecterror",
    "proxytimeouterror", "socksconnectionerror", "certificateverifyerror", "sslerror",
}

_READ_TIMEOUT_EXCEPTION_NAMES = {
    "readtimeout", "writetimeout", "servertimeouterror", "serverdisconnectederror",
}

_ALL_TRANSPORT_EXCEPTION_NAMES = _CONNECT_EXCEPTION_NAMES | _READ_TIMEOUT_EXCEPTION_NAMES | {
    "timeouterror", "timeout", "curlerror", "networkerror", "urlerror",
}


class UpstreamHTTPError(RuntimeError):
    """上游 HTTP 异常类"""

    def __init__(
        self,
        context: str,
        status_code: int,
        body: Any,
        retry_after: int | None = None,
    ) -> None:
        self.context = context
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        if isinstance(body, (dict, list)):
            try:
                body_str = json.dumps(body, ensure_ascii=False)
            except Exception:
                body_str = repr(body)
        else:
            body_str = str(body)
        if len(body_str) > _UPSTREAM_BODY_LOG_LIMIT:
            body_str = body_str[:_UPSTREAM_BODY_LOG_LIMIT] + "…[truncated]"
        super().__init__(f"{context} failed: status={status_code}, body={body_str}")


class ImageQuotaExceededError(RuntimeError):
    """ChatGPT 网页端以普通文本提示图像额度耗尽时使用的异常。"""

    def __init__(self, message: str, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"[image_quota_exhausted] {message}")


class QuotaExhaustedError(RuntimeError):
    """GPT 与 Gemini 文本分析/生图返回额度重置类提示时的通用异常。

    str(异常) 返回原始提示文本，供账号池 _classify_error 识别并冷却账号。
    """

    def __init__(self, message: str, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class ImageDownloadError(RuntimeError):
    """模型已完成生图，但结果图片下载重试后仍失败。"""


def is_connect_error(error: BaseException) -> bool:
    """
    判断是否为明确的 DNS、TCP 握手、代理连接建联阶段异常（此时上游未处理任务，可安全原地重试 1 次）。
    必须排除传输接收期截断（如 code=56 CURLE_RECV_ERROR、code=52 CURLE_GOT_NOTHING、ConnectionResetError）。
    """
    current: BaseException | None = error
    while current is not None:
        code = getattr(current, "code", None)
        msg = str(current).lower()
        name = type(current).__name__.lower()

        # 1. 明确的读取/接收期截断、数据未读全或服务端重置连接，严禁判定为建联错误（避免重发生图）
        if code in {52, 55, 56, 18, 8}:
            return False
        if isinstance(current, ConnectionResetError) or any(kw in msg for kw in ("reset by peer", "got nothing", "recv failure", "connection reset")):
            return False

        # 2. 明确的 DNS / 连接拒绝 / 代理握手建联错误码
        if code in {5, 6, 7, 35, 96, 97}:
            return True

        # 3. 超时代码 28：仅当明确在连接建立握手期超时才算建联超时；若为模型思考或读取超时则不算
        if code == 28:
            if any(kw in msg for kw in ("connection timed out", "connect() timed out", "failed to connect", "connecting to")):
                return True
            return False

        # 4. 异常类名判定（已排除上述接收截断）
        if name in {"connecttimeout", "dnserror", "proxyerror", "proxyconnecterror", "certificateverifyerror", "sslerror"}:
            return True
        if name in {
            "connecterror", "clientconnectorerror", "clientconnectordnserror",
            "clientconnectorcertificateerror", "clienthttpproxyerror", "clientoserror",
            "clientproxyconnectionerror", "proxyconnectionerror", "proxytimeouterror", "socksconnectionerror",
        }:
            return True
        if name in {"connectionerror", "clientconnectionerror"}:
            return True

        current = current.__cause__ or current.__context__
    return False


def is_read_timeout_error(error: BaseException) -> bool:
    """判断是否为请求已提交后的读取超时（Read Timeout / Thinking Timeout，严禁重发 POST 避免任务重复）。"""
    if is_connect_error(error):
        return False
    current: BaseException | None = error
    while current is not None:
        name = type(current).__name__.lower()
        if name in _READ_TIMEOUT_EXCEPTION_NAMES or name.endswith("readtimeout") or isinstance(current, (TimeoutError, asyncio.TimeoutError)):
            return True
        code = getattr(current, "code", None)
        if code == 28:
            return True
        if "timeout" in name or "timedout" in name:
            return True
        msg = str(current).lower()
        if ("operation timed out" in msg or "read timed out" in msg or "timed out" in msg) and not is_connect_error(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_transport_error(error: BaseException) -> bool:
    """判断异常链是否明确属于网络传输层（建联或读取），避免把代理/网络故障累计到账户健康度。"""
    if is_connect_error(error) or is_read_timeout_error(error):
        return True
    current: BaseException | None = error
    while current is not None:
        code = getattr(current, "code", None)
        if code in {52, 55, 56, 18, 8, 28, 5, 6, 7, 35, 96, 97}:
            return True
        name = type(current).__name__.lower()
        if name in _ALL_TRANSPORT_EXCEPTION_NAMES or name.endswith(("connecterror", "connecttimeout", "readtimeout")):
            return True
        if isinstance(current, ConnectionResetError):
            return True
        current = current.__cause__ or current.__context__
    return False


def parse_retry_after(body: Any, headers: Any = None) -> int | None:
    """
    统一解析上游返回的限流/额度重置等待时间 (秒)。
    优先解析 Header Retry-After，其次正则匹配中英文响应体中的时间描述。
    """
    if headers is not None and hasattr(headers, "get"):
        ra_header = headers.get("Retry-After")
        if ra_header is not None:
            ra_str = str(ra_header).strip()
            if ra_str.isdigit():
                return int(ra_str)

    body_str = str(body) if not isinstance(body, (dict, list)) else json.dumps(body, ensure_ascii=False)
    # 中文解析: 如 "已达到文件上传上限。1小时后再试。" 或 "在 2 小时 30 分钟后重试"
    match_cn = re.search(r"(?:将在|在)?\s*(\d+)\s*小时(?:\s*(\d+)\s*分钟)?(?:后再试|后重置|可继续)", body_str)
    if match_cn:
        hours = int(match_cn.group(1))
        minutes = int(match_cn.group(2) or 0)
        return hours * 3600 + minutes * 60

    # 英文解析: 如 "Try again in 1 hour" 或 "resets in 2 hours 15 minutes"
    match_en = re.search(r"(?:try again in|resets? in)\s*(\d+)\s*hour(?:s)?(?:\s*(\d+)\s*minute(?:s)?)?", body_str, re.IGNORECASE)
    if match_en:
        hours = int(match_en.group(1))
        minutes = int(match_en.group(2) or 0)
        return hours * 3600 + minutes * 60

    match_min = re.search(r"(?:将在|在|try again in|resets? in)\s*(\d+)\s*(?:分钟|minutes?)(?:后再试|后重置)?", body_str, re.IGNORECASE)
    if match_min:
        return int(match_min.group(1)) * 60

    return None


QUOTA_RESET_DEFAULT_SECONDS = 5 * 3600

# 单独出现即判定额度问题的硬性标记（无需配合状态词）
_HARD_QUOTA_MARKERS = (
    "image_quota_exhausted", "insufficient_quota", "upload_protection_limit_exceeded",
    "文件上传上限", "图片生成请求上限", "图像生成请求上限", "image generation limit",
    "usage limit", "usagelimitexceeded",
)
# 额度主题词与状态词需同时出现才判定，避免误伤普通文本
_QUOTA_TOPIC_WORDS = (
    "额度", "配额", "quota", "credits", "创建更多图片", "生成更多图片",
    "图像生成", "图片生成", "image generation",
)
_QUOTA_STATE_WORDS = (
    "重置", "耗尽", "不足", "用尽", "恢复", "使用情况", "设置",
    "resets", "reset", "exhausted", "reached", "exceeded", "insufficient", "limit",
)


def detect_quota_exhausted(text: Any, *, default_seconds: int | None = None) -> tuple[bool, int]:
    """
    识别额度耗尽 / 重置类表述，供 GPT 与 Gemini 统一使用。

    返回 (是否额度问题, 恢复秒数)：
    - 命中额度表述时优先提取网页给出的恢复时间；
    - 无法提取时使用默认冷却（缺省 5 小时 = 18000 秒）。
    - 合法的 JSON 输出不做判定。
    """
    if text is None:
        return False, 0
    body = str(text).strip()
    if not body:
        return False, 0
    # JSON 属于结构化合法输出，不是额度提示
    if (body.startswith("{") and body.endswith("}")) or (body.startswith("[") and body.endswith("]")):
        return False, 0

    lower = body.lower()
    if any(marker in lower for marker in _HARD_QUOTA_MARKERS):
        return True, parse_retry_after(body) or (
            default_seconds if default_seconds is not None else QUOTA_RESET_DEFAULT_SECONDS
        )

    has_topic = any(word in lower for word in _QUOTA_TOPIC_WORDS)
    has_state = any(word in lower for word in _QUOTA_STATE_WORDS)
    if has_topic and has_state:
        return True, parse_retry_after(body) or (
            default_seconds if default_seconds is not None else QUOTA_RESET_DEFAULT_SECONDS
        )
    return False, 0


def ensure_ok(response: requests.Response, context: str) -> None:
    if 200 <= response.status_code < 300:
        return
    body: Any = response.text
    try:
        body = response.json()
    except Exception:
        pass
    retry_after = parse_retry_after(body, getattr(response, "headers", None))
    raise UpstreamHTTPError(context, response.status_code, body, retry_after=retry_after)


def iter_sse_payloads(response: requests.Response) -> Iterator[str]:
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload:
            yield payload


async def aiter_sse_payloads(response: requests.Response) -> AsyncIterator[str]:
    """按完整 SSE 事件解析；跨网络块的行与多行 data 均保留。"""
    import codecs
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    data_lines: list[str] = []
    async for chunk in response.aiter_content():
        pending += decoder.decode(chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
    pending += decoder.decode(b"", final=True)
    # 未闭合事件表示正文截断，不能把它当作正常完成。
    if pending or data_lines:
        raise RuntimeError("Upstream SSE ended in an incomplete event")


def parse_cookie_string(raw: str) -> dict[str, str]:
    """Unified cookie string parser.
    Supports: standard Header format, JSON dict, JSON array, Netscape cookies.txt.
    """
    raw = raw.strip()
    # Try JSON first (dict or array)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("cookies", parsed.get("cookie", parsed))
        if isinstance(parsed, dict):
            return {str(k).removeprefix("HttpOnly_"): str(v) for k, v in parsed.items() if v}
        if isinstance(parsed, list):
            return {
                str(item.get("name", "")).removeprefix("HttpOnly_"): str(item.get("value", ""))
                for item in parsed
                if isinstance(item, dict) and item.get("name") and item.get("value")
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: standard header or Netscape cookies.txt
    cookies: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:  # Netscape cookies.txt format
            pairs = [(parts[5], parts[6])]
        else:
            pairs = [part.split("=", 1) for part in line.split(";") if "=" in part]
        for pair in pairs:
            if len(pair) == 2:
                name = pair[0].strip().removeprefix("HttpOnly_")
                value = pair[1].strip()
                if name and value:
                    cookies[name] = value
    return cookies
