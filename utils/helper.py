from __future__ import annotations
import json
from typing import Any, Iterator
from curl_cffi import requests

_UPSTREAM_BODY_LOG_LIMIT = 500
_TRANSPORT_EXCEPTION_NAMES = {
    "connectionerror", "connecterror", "connecttimeout", "readtimeout", "writetimeout",
    "timeouterror", "timeout", "curlerror", "certificateverifyerror", "sslerror", "dnserror",
    "clientconnectionerror", "clientconnectorerror", "clientconnectordnserror",
    "clientconnectorcertificateerror", "clienthttpproxyerror", "clientoserror",
    "clientproxyconnectionerror", "proxyerror", "proxyconnectionerror", "proxyconnecterror",
    "proxytimeouterror", "socksconnectionerror", "serverdisconnectederror", "servertimeouterror",
    "networkerror", "urlerror",
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


def is_transport_error(error: BaseException) -> bool:
    """判断异常链是否明确属于网络传输层，避免把代理故障累计到账户健康度。"""
    current: BaseException | None = error
    while current is not None:
        name = type(current).__name__.lower()
        if name in _TRANSPORT_EXCEPTION_NAMES or name.endswith(("connecterror", "connecttimeout", "readtimeout")):
            return True
        current = current.__cause__ or current.__context__
    return False


def ensure_ok(response: requests.Response, context: str) -> None:
    if 200 <= response.status_code < 300:
        return
    body: Any = response.text
    try:
        body = response.json()
    except Exception:
        pass
    retry_after_header = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    retry_after: int | None = None
    if retry_after_header is not None:
        ra_str = str(retry_after_header).strip()
        if ra_str.isdigit():
            retry_after = int(ra_str)
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
