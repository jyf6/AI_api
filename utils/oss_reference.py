from __future__ import annotations
import os
from functools import lru_cache
import threading
import time
import uuid
import io
import ipaddress
import socket
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener
from PIL import Image
import oss2
from utils.image_binary import image_media_type
from utils.image_stitch import _ensure_rgb, _encode_bounded_jpeg

# 模型附件使用十进制 MB，参考图先在 OSS 缩放，禁止回退下载原图。
MAX_REFERENCE_BYTES = 3_000_000
_CACHE_MAX_SIZE = 128
_CACHE_TTL = 900
_CACHE_MAX_BYTES = int(os.getenv("OSS_REFERENCE_CACHE_BYTES", "96000000"))
_cache_lock = threading.Lock()
_cache_data: dict[tuple, dict] = {}
_memory_data: dict[str, bytes] = {}
_flight_locks: dict[tuple, list] = {}

# 来源 URL 不依赖域名白名单，读取及每次重定向均检查解析地址。
_MAX_REMOTE_BYTES = 20 * 1024 * 1024


def _validate_reference_url(source: str) -> str:
    """完整 URL 保留查询参数，不将其他桶或亚马逊路径当作本桶对象键。"""
    url = urlsplit(source)
    if (url.scheme not in ("http", "https") or not url.hostname
            or url.username is not None or url.password is not None):
        raise ValueError("Reference image must use a valid HTTP/HTTPS URL")
    # 访问属性以验证端口语法，不限制公网图片服务使用其他端口。
    url.port
    return source


def _check_reference_address(source: str) -> None:
    """拒绝本机、内网和链路本地地址；图片域名无需事先登记。"""
    url = urlsplit(source)
    for address in socket.getaddrinfo(url.hostname, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(address[4][0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        private = (any(ip in network for network in (ipaddress.ip_network("10.0.0.0/8"),
                   ipaddress.ip_network("172.16.0.0/12"), ipaddress.ip_network("192.168.0.0/16")))
                   if isinstance(ip, ipaddress.IPv4Address) else ip in ipaddress.ip_network("fc00::/7") or ip.is_site_local)
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast or private:
            raise ValueError("Reference image cannot access local or private addresses")


class _ReferenceRedirectHandler(HTTPRedirectHandler):
    """允许跨域重定向，但重定向目标同样不能指向本机或内网。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_reference_url(newurl)
        _check_reference_address(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_url_reference(source: str, max_dimension: int) -> bytes:
    """按需读取到内存并压缩，源图不写文件、不上传 OSS；模型附件仍严格小于3MB。"""
    request = Request(_validate_reference_url(source), headers={"User-Agent": "Mozilla/5.0"})
    _check_reference_address(source)
    with build_opener(_ReferenceRedirectHandler()).open(request, timeout=15) as response:
        data = response.read(_MAX_REMOTE_BYTES + 1)
    if len(data) > _MAX_REMOTE_BYTES:
        raise ValueError("Remote reference image exceeds 20 MB")
    if image_media_type(data) is None:
        raise ValueError("Remote reference is not a valid image")
    with Image.open(io.BytesIO(data)) as image:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        rgb = _ensure_rgb(image).convert("RGB")
        try:
            return _encode_bounded_jpeg(rgb)
        finally:
            rgb.close()

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
    """分析、生图、调整共用 objectKey/URL 入口；规格参与缓存键，临时拼图独立保管。"""
    key = object_key.strip()
    if key.startswith("memory://"):
        with _cache_lock:
            return _memory_data[key]
    is_url = key.lower().startswith(("http://", "https://"))
    if is_url:
        _validate_reference_url(key)
    elif not key or key.startswith("/") or ":" in key:
        raise ValueError("Reference image must be an OSS objectKey")
    cache_key = (key, max_dimension, "jpg-v3")
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
            data = _fetch_url_reference(key, max_dimension) if is_url else _fetch_processed(key, max_dimension)
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
