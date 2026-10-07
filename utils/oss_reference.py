from __future__ import annotations

import os
from functools import lru_cache
import threading
import time
import uuid

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

_cache_lock = threading.Lock()
_cache_data: dict[str, dict] = {}
_CACHE_MAX_SIZE = 128
_CACHE_TTL = 900  # 15 minutes

# Singleflight pattern
_flight_locks: dict[str, threading.Lock] = {}

def read_oss_reference(object_key: str) -> bytes:
    """模型请求中的图片参数是 objectKey，禁止把外部 URL 当作对象键读取。"""
    key = object_key.strip()
    if key.startswith("memory://"):
        # retrieve from memory cache (stitched image)
        with _cache_lock:
            entry = _cache_data.get(key)
            if entry and time.monotonic() - entry["time"] < _CACHE_TTL:
                return entry["data"]
        raise ValueError(f"Memory reference {key} not found or expired")
        
    if not key or key.startswith("/") or key.lower().startswith(("http://", "https://", "data:")):
        raise ValueError("Reference image must be an OSS objectKey")
        
    with _cache_lock:
        entry = _cache_data.get(key)
        if entry and time.monotonic() - entry["time"] < _CACHE_TTL:
            entry["time"] = time.monotonic()  # update access time
            return entry["data"]
            
        flight_lock = _flight_locks.get(key)
        if flight_lock is None:
            flight_lock = threading.Lock()
            _flight_locks[key] = flight_lock
            
    with flight_lock:
        # Check cache again inside lock
        with _cache_lock:
            entry = _cache_data.get(key)
            if entry and time.monotonic() - entry["time"] < _CACHE_TTL:
                return entry["data"]
                
        # Actually fetch
        data = _bucket().get_object(key).read()
        
        with _cache_lock:
            # evict if full
            if len(_cache_data) >= _CACHE_MAX_SIZE:
                oldest_key = min(_cache_data.keys(), key=lambda k: _cache_data[k]["time"])
                del _cache_data[oldest_key]
            _cache_data[key] = {"data": data, "time": time.monotonic()}
            if key in _flight_locks:
                del _flight_locks[key]
                
        return data

def put_memory_reference(data: bytes) -> str:
    key = f"memory://{uuid.uuid4().hex}.jpg"
    with _cache_lock:
        if len(_cache_data) >= _CACHE_MAX_SIZE:
            oldest_key = min(_cache_data.keys(), key=lambda k: _cache_data[k]["time"])
            del _cache_data[oldest_key]
        _cache_data[key] = {"data": data, "time": time.monotonic()}
    return key
