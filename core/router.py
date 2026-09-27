from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

@dataclass(frozen=True)
class ResolvedModel:
    platform: str
    model: str = "auto"


def resolve_model(model_tag: str, expected_action: str) -> ResolvedModel:
    try:
        platform, action = model_tag.split("-", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="模型标签必须为 platform-action") from exc
    if action != expected_action or platform not in {"gpt", "gemini", "doubao"}:
        raise HTTPException(status_code=400, detail="模型平台或动作类型不匹配")
    return ResolvedModel(platform=platform, model="auto")
