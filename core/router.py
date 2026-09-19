from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from core.database import database


@dataclass(frozen=True)
class ResolvedModel:
    platform: str
    model: str


def resolve_model(model_tag: str, expected_action: str) -> ResolvedModel:
    try:
        platform, action = model_tag.split("-", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="模型标签必须为 platform-action") from exc
    if action != expected_action or platform not in {"gpt", "gemini", "doubao"}:
        raise HTTPException(status_code=400, detail="模型平台或动作类型不匹配")
    model = database.get_model(platform, expected_action)
    if not model:
        raise HTTPException(status_code=503, detail=f"未配置 {platform}-{expected_action} 模型")
    return ResolvedModel(platform, model)
