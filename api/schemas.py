from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from utils.image_ratio import parse_and_normalize_ratio


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="生成图片的提示词")
    model: str = "gpt-image"
    aspect_ratio: str = Field(..., description="约分后的宽高比，如 3:4")
    images: list[str] = Field(default_factory=list)

    @field_validator("aspect_ratio")
    @classmethod
    def validate_and_normalize_aspect_ratio(cls, v: str) -> str:
        _, _, normalized = parse_and_normalize_ratio(v)
        return normalized


class OAuthCallbackRequest(BaseModel):
    callback_url: str = Field(..., min_length=1, description="浏览器授权后跳转的完整 URL 或 Code")
    session_id: str = Field(default="", description="OAuth 会话 ID")
    proxy: str = Field(default="", description="可选独立代理节点 (http://user:pass@ip:port 或 socks5://...)")


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-chat"
    prompt: str = Field(..., min_length=1)
    images: list[str] = Field(default_factory=list)


class DoubaoAccountRequest(BaseModel):
    name: str = ""
    cookie: str = Field(..., min_length=1)
    proxy: str = ""


class GeminiAccountRequest(BaseModel):
    name: str = ""
    cookie: str = Field(..., min_length=1)
    proxy: str = ""


class AccountCookieUpdateRequest(BaseModel):
    """仅更新账号 Cookie，代理节点由服务端保留原值。"""

    cookie: str = Field(..., min_length=1)


