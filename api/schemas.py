from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from utils.image_ratio import parse_and_normalize_ratio


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="生成图片的提示词")
    model: str = "gpt-image"
    aspect_ratio: str = Field(..., description="约分后的宽高比，如 3:4")
    images: list[str] = Field(default_factory=list, description="参考图的阿里云 OSS objectKey 列表")
    request_id: str = Field(default="", max_length=100, pattern=r"^[A-Za-z0-9._:-]*$")
    # Java 对同一 request_id 重发时的序号；其他调用方可以不传。
    java_attempt: int | None = Field(default=None, ge=1)
    dispatch_id: int | None = None
    task_code: str = Field(default="", max_length=100, pattern=r"^[A-Za-z0-9._:-]*$")
    operation_id: int | None = None
    item_id: int | None = None
    stage: str = Field(default="", max_length=80, pattern=r"^[A-Za-z0-9._:-]*$")

    @field_validator("aspect_ratio")
    @classmethod
    def validate_and_normalize_aspect_ratio(cls, v: str) -> str:
        _, _, normalized = parse_and_normalize_ratio(v)
        return normalized


class OAuthCallbackRequest(BaseModel):
    callback_url: str = Field(..., min_length=1, description="浏览器授权后跳转的完整 URL 或 Code")
    session_id: str = Field(default="", description="OAuth 会话 ID")
    proxy: str = Field(default="", description="可选独立代理节点 (http://user:pass@ip:port 或 socks5://...)")
    proxy_id: int | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-chat"
    prompt: str = Field(..., min_length=1)
    images: list[str] = Field(default_factory=list, description="参考图的阿里云 OSS objectKey 列表")
    request_id: str = Field(default="", max_length=100, pattern=r"^[A-Za-z0-9._:-]*$")
    java_attempt: int | None = Field(default=None, ge=1)
    dispatch_id: int | None = None
    task_code: str = Field(default="", max_length=100, pattern=r"^[A-Za-z0-9._:-]*$")
    operation_id: int | None = None
    item_id: int | None = None
    stage: str = Field(default="", max_length=80, pattern=r"^[A-Za-z0-9._:-]*$")


class DoubaoAccountRequest(BaseModel):
    name: str = ""
    cookie: str = Field(..., min_length=1)
    proxy: str = ""
    proxy_id: int | None = None


class GeminiAccountRequest(BaseModel):
    name: str = ""
    cookie: str = Field(..., min_length=1)
    proxy: str = ""
    proxy_id: int | None = None


class ProxyNodeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    proxy_url: str = Field(..., min_length=1, max_length=1000)
    status: Literal["active", "disabled"] = "active"

    @field_validator("proxy_url")
    @classmethod
    def validate_proxy_url(cls, value: str) -> str:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in {"http", "https", "socks5", "socks5h"} or not parts.hostname or not parts.port:
            raise ValueError("代理地址必须是包含主机和端口的 HTTP(S) 或 SOCKS5 URL")
        return value.strip()


class ProxyBindingRequest(BaseModel):
    proxy_id: int | None = None


class ProxyStatusRequest(BaseModel):
    status: Literal["active", "disabled"]


class AccountCookieUpdateRequest(BaseModel):
    """仅更新账号 Cookie，代理节点由服务端保留原值。"""

    cookie: str = Field(..., min_length=1)


