from __future__ import annotations
import os
from functools import lru_cache
import threading
import time
import uuid
import oss2
from utils.image_binary import image_media_type

# 模型附件使用十进制 MB，参考图先在 OSS 缩放，禁止回退下载原图。
MAX_REFERENCE_BYTES = 3_000_000
_CACHE_MAX_SIZE = 128
_CACHE_TTL = 900
_CACHE_MAX_BYTES = int(os.getenv("OSS_REFERENCE_CACHE_BYTES", "96000000"))
_cache_lock = threading.Lock()
_cache_data: dict[tuple, dict] = {}
_memory_data: dict[str, bytes] = {}
_flight_locks: dict[tuple, list] = {}

@lru_cache(maxsize=1)
def _bucket() -> oss2.Bucket:
    """使用部署指定的 bucket，不使用模型账号代理。"""
    return oss2.Bucket(oss2.Auth(os.environ["OSS_ACCESS_ID"], os.environ["OSS_ACCESS_KEY"]),
                       os.environ["OSS_ENDPOINT"], os.environ["OSS_BUCKET"])

def _fetch_processed(key: str, max_dimension: int) -> bytes:
    """只读取 OSS 处理结果；超限即关闭响应，缩小规格后重新请求。"""
    dimension, quality = max_dimension, 85
    for _ in range(8):
        process = (f"image/resize,m_lfit,w_{dimension},h_{dimension},limit_1"
                   f"/format,jpg/quality,q_{quality}")
        response = _bucket().get_object(key, process=process)
        try:
            chunks, size = [], 0
            while True:
                chunk = response.read(min(65536, MAX_REFERENCE_BYTES - size))
                if not chunk:
                    data = b"".join(chunks)
                    if image_media_type(data) is None:
                        raise ValueError("OSS processed reference is not a valid image")
                    return data
                size += len(chunk)
                if size >= MAX_REFERENCE_BYTES:
                    break
                chunks.append(chunk)
        finally:
            response.close()
        if quality > 65:
            quality = max(65, quality - 10)
        else:
            dimension = max(256, int(dimension * 0.75))
    raise ValueError("OSS reference cannot be reduced below 3 MB")

def read_oss_reference(object_key: str, max_dimension: int = 2048) -> bytes:
    """分析、生图、调整共用入口，规格参与缓存键，临时拼图独立保管。"""
    key = object_key.strip()
    if key.startswith("memory://"):
        with _cache_lock:
            return _memory_data[key]
    if not key or key.startswith("/") or key.lower().startswith(("http://", "https://", "data:")):
        raise ValueError("Reference image must be an OSS objectKey")
    cache_key = (key, max_dimension, "jpg-v2")
    with _cache_lock:
        entry = _cache_data.get(cache_key)
        if entry and time.monotonic() - entry["time"] < _CACHE_TTL:
            entry["time"] = time.monotonic()
            return entry["data"]
        flight = _flight_locks.setdefault(cache_key, [threading.Lock(), 0])
        flight[1] += 1
    try:
        with flight[0]:
            with _cache_lock:
                entry = _cache_data.get(cache_key)
                if entry and time.monotonic() - entry["time"] < _CACHE_TTL:
                    return entry["data"]
            data = _fetch_processed(key, max_dimension)
            with _cache_lock:
                now = time.monotonic()
                for expired in [k for k, v in _cache_data.items() if now - v["time"] >= _CACHE_TTL]:
                    del _cache_data[expired]
                while _cache_data and (len(_cache_data) >= _CACHE_MAX_SIZE or
                        sum(len(v["data"]) for v in _cache_data.values()) + len(data) > _CACHE_MAX_BYTES):
                    del _cache_data[min(_cache_data, key=lambda k: _cache_data[k]["time"])]
                if len(data) <= _CACHE_MAX_BYTES:
                    _cache_data[cache_key] = {"data": data, "time": now}
            return data
    finally:
        with _cache_lock:
            flight[1] -= 1
            if not flight[1]:
                del _flight_locks[cache_key]

def put_memory_reference(data: bytes) -> str:
    """活动拼图不参与普通缓存淘汰，由请求 finally 释放。"""
    if len(data) >= MAX_REFERENCE_BYTES or image_media_type(data) is None:
        raise ValueError("Model attachment must be a valid image below 3 MB")
    key = f"memory://{uuid.uuid4().hex}.jpg"
    with _cache_lock:
        _memory_data[key] = data
    return key

def release_memory_references(keys: list[str]) -> None:
    """成功、失败及取消时都释放请求专属图片。"""
    with _cache_lock:
        for key in keys:
            _memory_data.pop(key, None)
