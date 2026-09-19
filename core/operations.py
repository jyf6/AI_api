from __future__ import annotations

import uuid

from fastapi import HTTPException, Response

from core.database import database


def begin_operation(operation_id: str | None, action: str) -> tuple[str, dict | None]:
    """创建一次可恢复调用；相同操作 ID 只返回已保存结果，不会再次提交上游。"""
    resolved_id = operation_id or str(uuid.uuid4())
    operation = database.get_operation(resolved_id)
    if operation is None:
        if database.create_operation(resolved_id, action):
            return resolved_id, None
        # 并发请求可能在查询与插入之间创建了同一操作，重新读取后只复用原执行。
        operation = database.get_operation(resolved_id)
    if operation["action"] != action:
        raise HTTPException(status_code=409, detail="操作标识与调用类型不匹配")
    return resolved_id, operation


def replay_image(operation: dict, operation_id: str) -> Response | None:
    if operation["status"] == "SUCCEEDED" and operation["image_result"]:
        return Response(content=operation["image_result"], media_type=operation["content_type"] or "image/png", headers={"X-Operation-Id": operation_id})
    if operation["status"] != "RUNNING":
        raise HTTPException(status_code=409, detail="上游结果待恢复，请使用相同操作标识稍后查询")
    raise HTTPException(status_code=202, detail="操作正在执行，请使用相同操作标识稍后查询")


def replay_text(operation: dict, operation_id: str) -> dict | None:
    if operation["status"] == "SUCCEEDED" and operation["text_result"]:
        return {"code": 0, "text": operation["text_result"], "operation_id": operation_id}
    if operation["status"] != "RUNNING":
        raise HTTPException(status_code=409, detail="上游结果待恢复，请使用相同操作标识稍后查询")
    raise HTTPException(status_code=202, detail="操作正在执行，请使用相同操作标识稍后查询")
