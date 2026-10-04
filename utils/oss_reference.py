from __future__ import annotations

import os
from functools import lru_cache

import oss2


@lru_cache(maxsize=1)
def _bucket() -> oss2.Bucket:
    """参考图只从部署配置指定的阿里云 OSS bucket 读取。"""
    access_id = os.environ["OSS_ACCESS_ID"]
    access_key = os.environ["OSS_ACCESS_KEY"]
    if not access_id or not access_key:
        raise RuntimeError("OSS reference credentials are not configured")
    return oss2.Bucket(
        oss2.Auth(access_id, access_key),
        os.environ["OSS_ENDPOINT"],
        os.environ["OSS_BUCKET"],
    )


def read_oss_reference(object_key: str) -> bytes:
    """模型请求中的图片参数是 objectKey，禁止把外部 URL 当作对象键读取。"""
    key = object_key.strip()
    if not key or key.startswith("/") or key.lower().startswith(("http://", "https://", "data:")):
        raise ValueError("Reference image must be an OSS objectKey")
    return _bucket().get_object(key).read()
