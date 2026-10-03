from __future__ import annotations

import logging
import hashlib

logger = logging.getLogger("chatgpt-image-service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
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
