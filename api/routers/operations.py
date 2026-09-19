from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from core.database import database

router = APIRouter(tags=["operations"])


@router.get("/v1/operations/{operation_id}")
def get_operation(operation_id: str):
    """查询同一操作的恢复状态；图片成功后直接返回已保存二进制结果。"""
    operation = database.get_operation(operation_id)
    if operation is None:
        raise HTTPException(status_code=404, detail="操作不存在")
    if operation["action"] == "image" and operation["status"] == "SUCCEEDED":
        return Response(content=operation["image_result"], media_type=operation["content_type"] or "image/png", headers={"X-Operation-Id": operation_id})
    return {
        "operation_id": operation_id,
        "action": operation["action"],
        "status": operation["status"],
        "text": operation["text_result"] if operation["status"] == "SUCCEEDED" else None,
        "error": operation["error_message"],
    }
