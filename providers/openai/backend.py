from __future__ import annotations
import hashlib
import urllib.request
import uuid

import json
import re
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional
from curl_cffi import requests

from utils.helper import ImageQuotaExceededError, ensure_ok, iter_sse_payloads
from utils.log import logger
from utils.image_binary import image_media_type
from utils.pow import build_legacy_requirements_token, build_proof_token, parse_pow_resources
from utils.turnstile import solve_turnstile_token

DEFAULT_CLIENT_VERSION = "prod-a194cd50d4416d3c0b47c740f206b12ce60f5887"
DEFAULT_CLIENT_BUILD_NUMBER = "6708908"

REAL_IMAGE_FILE_ID_RE = re.compile(r"\bfile_00000000[a-f0-9]{24}\b")
SEDIMENT_ID_RE = re.compile(r"sediment://([A-Za-z0-9_-]+)")
FILE_SERVICE_ID_RE = re.compile(r"file-service://([A-Za-z0-9_-]+)")

# ── Bootstrap cache (module-level, shared across all instances) ──
_bootstrap_cache_lock = threading.Lock()
_cached_pow_scripts: list[str] = []
_cached_data_build: str = ""
_cache_timestamp: float = 0.0
_CACHE_TTL: float = 1800.0  # 30 minutes


@dataclass
class ChatRequirements:
    token: str
    proof_token: str = ""
    turnstile_token: str = ""
    so_token: str = ""


class OpenAIBackendAPI:
    """ChatGPT reverse client with per-account proxy binding and zero-storage image output."""

    def __init__(self, access_token: str = "", proxy: str = "") -> None:
        self.access_token = access_token.strip()
        self.proxy = proxy.strip()
        self.base_url = "https://chatgpt.com"
        self.client_version = DEFAULT_CLIENT_VERSION
        self.client_build_number = DEFAULT_CLIENT_BUILD_NUMBER
        self.device_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
        self.sec_ch_ua = '"Chromium";v="145", "Not:A-Brand";v="99"'

        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        self.session = requests.Session(impersonate="chrome124", proxies=proxies)
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Sec-Ch-Ua": self.sec_ch_ua,
            "Sec-Ch-Ua-Arch": '"x86"',
            "Sec-Ch-Ua-Bitness": '"64"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "OAI-Device-Id": self.device_id,
            "OAI-Session-Id": self.session_id,
            "OAI-Language": "zh-CN",
            "OAI-Client-Version": self.client_version,
            "OAI-Client-Build-Number": self.client_build_number,
        })
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"

        self.pow_script_sources: list[str] = []
        self.pow_data_build: str = ""

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _headers(self, path: str, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        headers = dict(self.session.headers)
        headers["X-OpenAI-Target-Path"] = path
        headers["X-OpenAI-Target-Route"] = path
        if extra:
            headers.update(extra)
        return headers

    def _bootstrap(self) -> None:
        """Fetch homepage to extract PoW script URLs and data-build tag.
        Uses a 30-minute module-level cache to avoid redundant HTML fetches.
        """
        global _cached_pow_scripts, _cached_data_build, _cache_timestamp

        with _bootstrap_cache_lock:
            if _cached_pow_scripts and (time.time() - _cache_timestamp) < _CACHE_TTL:
                self.pow_script_sources = list(_cached_pow_scripts)
                self.pow_data_build = _cached_data_build
                logger.debug("Bootstrap: using cached PoW resources")
                return

        path = "/"
        response = self.session.get(
            self.base_url + path,
            headers=self._headers(path, {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}),
            timeout=30,
        )
        ensure_ok(response, "bootstrap")
        self.pow_script_sources, self.pow_data_build = parse_pow_resources(response.text)
        with _bootstrap_cache_lock:
            _cached_pow_scripts = list(self.pow_script_sources)
            _cached_data_build = self.pow_data_build
            _cache_timestamp = time.time()
            logger.debug("Bootstrap: refreshed PoW cache")

    def _get_chat_requirements(self) -> ChatRequirements:
        """Obtain Sentinel Token (PoW proof) required for image generation."""
        base = "/backend-api/sentinel/chat-requirements"
        p_token = build_legacy_requirements_token(self.user_agent, self.pow_script_sources, self.pow_data_build)

        prepare_path = base + "/prepare"
        response = self.session.post(
            self.base_url + prepare_path,
            headers=self._headers(prepare_path, {"Content-Type": "application/json"}),
            json={"p": p_token},
            timeout=30,
        )
        ensure_ok(response, "chat_requirements_prepare")
        prepare_data = response.json()

        proof_token = ""
        proof_info = prepare_data.get("proofofwork") or {}
        if proof_info.get("required"):
            proof_token = build_proof_token(
                proof_info.get("seed", ""),
                proof_info.get("difficulty", ""),
                self.user_agent,
                script_sources=self.pow_script_sources,
                data_build=self.pow_data_build,
            )

        turnstile_token = ""
        turnstile_info = prepare_data.get("turnstile") or {}
        if turnstile_info.get("required") and turnstile_info.get("dx"):
            turnstile_token = solve_turnstile_token(turnstile_info["dx"], p_token) or ""

        finalize_path = base + "/finalize"
        response = self.session.post(
            self.base_url + finalize_path,
            headers=self._headers(finalize_path, {"Content-Type": "application/json"}),
            json={
                "prepare_token": prepare_data.get("prepare_token", ""),
                "proof_token": proof_token,
                "turnstile_token": turnstile_token,
            },
            timeout=30,
        )
        ensure_ok(response, "chat_requirements_finalize")
        data = response.json()
        token = data.get("token", "")
        if not token:
            raise RuntimeError("Failed to obtain sentinel token from upstream")

        return ChatRequirements(
            token=token,
            proof_token=proof_token,
            turnstile_token=turnstile_token,
            so_token=data.get("so_token", ""),
        )

    def _image_headers(self, path: str, reqs: ChatRequirements, conduit_token: str = "", accept: str = "*/*") -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "OpenAI-Sentinel-Chat-Requirements-Token": reqs.token,
        }
        if reqs.proof_token:
            headers["OpenAI-Sentinel-Proof-Token"] = reqs.proof_token
        if conduit_token:
            headers["X-Conduit-Token"] = conduit_token
        if accept == "text/event-stream":
            headers["X-Oai-Turn-Trace-Id"] = str(uuid.uuid4())
        return self._headers(path, headers)

    def _conversation_headers(self, path: str, reqs: ChatRequirements) -> dict[str, str]:
        headers = self._headers(path, {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "OpenAI-Sentinel-Chat-Requirements-Token": reqs.token,
        })
        if reqs.proof_token:
            headers["OpenAI-Sentinel-Proof-Token"] = reqs.proof_token
        if reqs.turnstile_token:
            headers["OpenAI-Sentinel-Turnstile-Token"] = reqs.turnstile_token
        if reqs.so_token:
            headers["OpenAI-Sentinel-SO-Token"] = reqs.so_token
        return headers

    @staticmethod
    def _image_dimensions(data: bytes) -> tuple[int, int]:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        if data.startswith(b"\xff\xd8"):
            offset = 2
            while offset + 9 < len(data):
                if data[offset] != 0xFF:
                    offset += 1
                    continue
                marker = data[offset + 1]
                offset += 2
                if marker in (0xD8, 0xD9):
                    continue
                if offset + 2 > len(data):
                    break
                segment_len = struct.unpack(">H", data[offset:offset + 2])[0]
                if marker in range(0xC0, 0xC4) or marker in range(0xC5, 0xC8) or marker in range(0xC9, 0xCC) or marker in range(0xCD, 0xD0):
                    if offset + 7 <= len(data):
                        return struct.unpack(">HH", data[offset + 3:offset + 7])[::-1]
                offset += segment_len
        return 0, 0

    def _upload_image_data(self, value: str, file_name: str = "image.png") -> dict[str, Any]:
        data = urllib.request.urlopen(value, timeout=30).read()
        mime = image_media_type(data)
        if mime is None:
            raise RuntimeError("GPT reference download did not return a valid image")
        path = "/backend-api/files"
        width, height = self._image_dimensions(data)
        response = self.session.post(self.base_url + path, headers=self._headers(path, {"Content-Type": "application/json", "Accept": "application/json"}),
                                     json={"file_name": file_name, "file_size": len(data), "use_case": "multimodal", "width": width, "height": height}, timeout=60)
        ensure_ok(response, path)
        meta = response.json()
        response = self.session.put(meta["upload_url"], headers={"Content-Type": mime, "x-ms-blob-type": "BlockBlob", "x-ms-version": "2020-04-08"}, data=data, timeout=120)
        ensure_ok(response, "image_upload")
        path = f"/backend-api/files/{meta['file_id']}/uploaded"
        response = self.session.post(self.base_url + path, headers=self._headers(path, {"Content-Type": "application/json"}), data="{}", timeout=60)
        ensure_ok(response, path)
        return {"file_id": meta["file_id"], "file_name": file_name, "mime_type": mime, "file_size": len(data), "width": width, "height": height, "content_hash": hashlib.sha256(data).digest()}

    def _conversation_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for item in messages:
            role = str(item.get("role") or "user")
            content = item.get("content", "")
            if isinstance(content, str):
                parts: list[Any] = [content]
            else:
                parts = []
                uploads: list[dict[str, Any]] = []
                for part in content if isinstance(content, list) else []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                    elif part.get("type") in {"image_url", "image"}:
                        image_url = part.get("image_url")
                        value = image_url.get("url") if isinstance(image_url, dict) else part.get("data")
                        if not value:
                            continue
                        ref = self._upload_image_data(str(value))
                        uploads.append(ref)
                        parts.append({"content_type": "image_asset_pointer", "asset_pointer": f"file-service://{ref['file_id']}",
                                      "width": ref["width"], "height": ref["height"], "size_bytes": ref["file_size"]})
            attachments = []
            if uploads:
                pointers = [p for p in parts if isinstance(p, dict)]
                text_parts = [p for p in parts if isinstance(p, str)]
                parts = pointers + text_parts
                attachments = [{"id": ref["file_id"], "mimeType": ref["mime_type"], "name": ref["file_name"],
                                "size": ref["file_size"], "width": ref["width"], "height": ref["height"]}
                               for ref in uploads]
            message = {"id": str(uuid.uuid4()), "author": {"role": role},
                       "content": {"content_type": "multimodal_text" if attachments else "text", "parts": parts}}
            if attachments:
                message["metadata"] = {"attachments": attachments}
            converted.append(message)
        return converted

    def stream_chat(self, messages: list[dict[str, Any]], model: str = "gpt-5-5") -> Iterator[str]:
        self._bootstrap()
        reqs = self._get_chat_requirements()
        path = "/backend-api/conversation"
        payload = {"action": "next", "messages": self._conversation_messages(messages), "model": model, "parent_message_id": str(uuid.uuid4()),
                   "conversation_mode": {"kind": "primary_assistant"}, "conversation_origin": None,
                   "force_paragen": False, "force_paragen_model_slug": "", "force_rate_limit": False,
                   "force_use_sse": True, "history_and_training_disabled": True, "reset_rate_limits": False,
                   "suggestions": [], "supported_encodings": [], "system_hints": [], "timezone": "Asia/Shanghai",
                   "timezone_offset_min": -480, "variant_purpose": "comparison_implicit", "websocket_request_id": str(uuid.uuid4()),
                   "client_contextual_info": {"is_dark_mode": False, "time_since_loaded": 120,
                                               "page_height": 900, "page_width": 1400, "pixel_ratio": 2,
                                               "screen_height": 1440, "screen_width": 2560}}
        response = self.session.post(self.base_url + path, headers=self._conversation_headers(path, reqs), json=payload, timeout=300, stream=True)
        ensure_ok(response, path)
        try:
            yield from iter_sse_payloads(response)
        finally:
            response.close()

    def chat_text(self, prompt: str, images: list[str] | None = None, model: str = "gpt-5-5") -> str:
        text = ""
        messages = [{"role": "user", "content": ([{"type": "text", "text": prompt}] +
                    [{"type": "image_url", "image_url": {"url": url}} for url in (images or [])])}]
        for raw in self.stream_chat(messages, model):
            if raw == "[DONE]":
                break
            try:
                event = json.loads(raw)
            except Exception:
                continue
            for candidate in (event, event.get("v") if isinstance(event, dict) else None):
                if not isinstance(candidate, dict):
                    continue
                message = candidate.get("message") or {}
                if (message.get("author") or {}).get("role") != "assistant":
                    continue
                content = message.get("content") or {}
                text = content.get("text") or "" if isinstance(content, dict) else ""
                if not text and isinstance(content, dict):
                    text = "".join(str(part) for part in content.get("parts") or [] if isinstance(part, str))
        if not text.strip():
            raise RuntimeError("Upstream chat returned no assistant text")
        return text

    def _prepare_image_conversation(self, prompt: str, reqs: ChatRequirements, model: str = "gpt-5-5") -> str:
        """Prepare image conversation and obtain conduit_token."""
        path = "/backend-api/f/conversation/prepare"
        payload = {
            "action": "next",
            "fork_from_shared_post": False,
            "parent_message_id": str(uuid.uuid4()),
            "model": model,
            "client_prepare_state": "success",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": ["picture_v2"],
            "partial_query": {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
            },
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {"app_name": "chatgpt.com"},
        }
        response = self.session.post(
            self.base_url + path,
            headers=self._image_headers(path, reqs),
            json=payload,
            timeout=60,
        )
        ensure_ok(response, path)
        return response.json().get("conduit_token", "")

    def _start_image_generation(self, prompt: str, reqs: ChatRequirements, conduit_token: str, model: str = "gpt-5-5", references: list[str] | None = None) -> tuple[requests.Response, set[bytes], set[str]]:
        """Initiate image generation SSE long connection."""
        path = "/backend-api/f/conversation"
        parts: list[Any] = []
        attachments = []
        reference_hashes: set[bytes] = set()
        reference_file_ids: set[str] = set()
        if references:
            for value in references:
                if value:
                    ref = self._upload_image_data(value)
                    reference_hashes.add(ref["content_hash"])
                    reference_file_ids.add(ref["file_id"])
                    parts.append({"content_type": "image_asset_pointer", "asset_pointer": f"file-service://{ref['file_id']}",
                                  "width": ref["width"], "height": ref["height"], "size_bytes": ref["file_size"]})
                    attachments.append({"id": ref["file_id"], "mimeType": ref["mime_type"], "name": ref["file_name"],
                                        "size": ref["file_size"], "width": ref["width"], "height": ref["height"]})
        parts.append(prompt)
        
        message = {
            "id": str(uuid.uuid4()),
            "author": {"role": "user"},
            "create_time": time.time(),
            "content": {"content_type": "multimodal_text" if attachments else "text", "parts": parts},
            "metadata": {
                "developer_mode_connector_ids": [],
                "selected_github_repos": [],
                "selected_all_github_repos": False,
                "system_hints": ["picture_v2"],
                "serialization_metadata": {"custom_symbol_offsets": []},
            },
        }
        if attachments:
            message["metadata"]["attachments"] = attachments

        payload = {
            "action": "next",
            "messages": [message],
            "parent_message_id": str(uuid.uuid4()),
            "model": model,
            "client_prepare_state": "sent",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": True,
            "system_hints": ["picture_v2"],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "is_dark_mode": False,
                "time_since_loaded": 1200,
                "page_height": 1072,
                "page_width": 1724,
                "pixel_ratio": 1.2,
                "screen_height": 1440,
                "screen_width": 2560,
                "app_name": "chatgpt.com",
            },
            "paragen_cot_summary_display_override": "allow",
            "force_parallel_switch": "auto",
        }
        response = self.session.post(
            self.base_url + path,
            headers=self._image_headers(path, reqs, conduit_token, accept="text/event-stream"),
            json=payload,
            timeout=300,
            stream=True,
        )
        ensure_ok(response, path)
        return response, reference_hashes, reference_file_ids

    @staticmethod
    def _asset_ids_from_payload(payload: Any, excluded_file_ids: set[str] | None = None) -> tuple[list[str], list[str]]:
        """从结构化生图载荷中提取两类下载资产。"""
        file_ids: list[str] = []
        sediment_ids: list[str] = []
        excluded_file_ids = excluded_file_ids or set()

        def walk(value: Any) -> None:
            if isinstance(value, str):
                sediment_matches = SEDIMENT_ID_RE.findall(value)
                # sediment://file_xxx 中的标识属于会话附件，不能再作为文件服务资源重复下载。
                file_value = SEDIMENT_ID_RE.sub("", value)
                for file_id in REAL_IMAGE_FILE_ID_RE.findall(file_value) + FILE_SERVICE_ID_RE.findall(file_value):
                    if file_id not in excluded_file_ids and file_id not in file_ids:
                        file_ids.append(file_id)
                for sediment_id in sediment_matches:
                    if sediment_id not in sediment_ids:
                        sediment_ids.append(sediment_id)
            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)
        return file_ids, sediment_ids

    @classmethod
    def _message_image_asset_ids(
        cls, message: dict[str, Any], excluded_file_ids: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """仅从 GPT 生图消息提取资产，避免把用户上传参考图当作结果。"""
        role = str((message.get("author") or {}).get("role") or "").lower()
        metadata = message.get("metadata") or {}
        content = message.get("content") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        is_image_gen = metadata.get("async_task_type") == "image_gen"
        has_asset_pointer = bool(FILE_SERVICE_ID_RE.search(str(content))) or bool(SEDIMENT_ID_RE.search(str(content)))
        if role != "tool" or not is_image_gen or not has_asset_pointer:
            return [], []
        return cls._asset_ids_from_payload({"content": content, "metadata": metadata}, excluded_file_ids)

    @classmethod
    def _sse_image_asset_ids(
        cls, event: dict[str, Any], excluded_file_ids: set[str] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """标准化 /f/conversation 的完整消息与 Patch，提取即时生图结果。"""
        conversation_id = ""
        file_ids: list[str] = []
        sediment_ids: list[str] = []

        def collect(value: Any, inherited_conversation_id: str = "") -> None:
            nonlocal conversation_id
            if not isinstance(value, dict):
                return

            # 批量 Patch 的子项通常不带会话 ID，沿用其外层事件的会话上下文。
            current_conversation_id = str(value.get("conversation_id") or inherited_conversation_id or "")
            if current_conversation_id:
                conversation_id = current_conversation_id

            message = value.get("message")
            if isinstance(message, dict):
                found_files, found_sediments = cls._message_image_asset_ids(message, excluded_file_ids)
                for asset_id in found_files:
                    if asset_id not in file_ids:
                        file_ids.append(asset_id)
                for asset_id in found_sediments:
                    if asset_id not in sediment_ids:
                        sediment_ids.append(asset_id)

            # 上游既可能用 v.message 传完整消息，也可能在 v 数组内打包多个 Patch。
            patch_value = value.get("v")
            if isinstance(patch_value, dict):
                collect(patch_value, current_conversation_id)
            elif isinstance(patch_value, list):
                for patch in patch_value:
                    collect(patch, current_conversation_id)

        collect(event)
        return conversation_id, file_ids, sediment_ids

    @staticmethod
    def _sse_image_quota_error(event: dict[str, Any]) -> tuple[str, int] | None:
        """识别 GPT 生图 SSE 的非用户额度提示，并解析网页给出的实际等待时间。"""
        messages: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if not isinstance(value, dict):
                return
            message = value.get("message")
            if isinstance(message, dict):
                messages.append(message)
            patch_value = value.get("v")
            if isinstance(patch_value, dict):
                collect(patch_value)
            elif isinstance(patch_value, list):
                for patch in patch_value:
                    collect(patch)

        collect(event)
        for message in messages:
            if str((message.get("author") or {}).get("role") or "").lower() == "user":
                continue
            content = message.get("content") or {}
            if not isinstance(content, dict):
                continue
            text_parts = [str(content.get("text") or "")]
            text_parts.extend(str(part) for part in content.get("parts") or [] if isinstance(part, str))
            text = "\n".join(part for part in text_parts if part).strip()
            lower = text.lower()
            has_image = "图像生成" in text or "图片生成" in text or "image generation" in lower
            has_limit = "上限" in text or "额度" in text or "limit" in lower or "quota" in lower
            has_reset = "重置" in text or "后可" in text or "reset" in lower
            if not (has_image and has_limit and has_reset):
                continue

            chinese = re.search(r"(?:将在|在)\s*(\d+)\s*小时(?:\s*(\d+)\s*分钟)?后重置", text)
            english = re.search(r"resets?\s+in\s+(\d+)\s+hours?(?:\s+and\s+(\d+)\s+minutes?)?", lower)
            match = chinese or english
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2) or 0)
                # 冷却比网页展示时间多 30 秒，避免边界时刻再次命中额度限制。
                return text, hours * 3600 + minutes * 60 + 30
            # 未携带具体重置时间时短暂冷却后再探测，不能伪造固定的长额度窗口。
            return text, 300
        return None

    @classmethod
    def _generated_image_asset_ids(
        cls, mapping: dict[str, Any], excluded_file_ids: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """提取会话 mapping 中 GPT 生图工具输出的可下载资产。"""
        file_ids: list[str] = []
        sediment_ids: list[str] = []
        for node in mapping.values():
            found_files, found_sediments = cls._message_image_asset_ids(
                ((node or {}).get("message") or {}), excluded_file_ids,
            )
            for asset_id in found_files:
                if asset_id not in file_ids:
                    file_ids.append(asset_id)
            for asset_id in found_sediments:
                if asset_id not in sediment_ids:
                    sediment_ids.append(asset_id)
        return file_ids, sediment_ids

    @classmethod
    def _generated_image_file_ids(cls, mapping: dict[str, Any], excluded_file_ids: set[str] | None = None) -> list[str]:
        """兼容原有轮询调用：仅返回可通过文件接口下载的生成资产。"""
        return cls._generated_image_asset_ids(mapping, excluded_file_ids)[0]

    @classmethod
    def _conversation_generated_image_asset_ids(
        cls, mapping: dict[str, Any], excluded_file_ids: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """按 ChatGPT 会话返回格式提取结果，兼容落盘后缺失工具元数据的消息。"""
        file_ids, sediment_ids = cls._generated_image_asset_ids(mapping, excluded_file_ids)
        if file_ids or sediment_ids:
            return file_ids, sediment_ids

        # 对齐成熟客户端的会话轮询：图片已生成但工具元数据被上游裁剪时，扫描非用户消息的资源指针。
        for node in mapping.values():
            message = ((node or {}).get("message") or {})
            role = str((message.get("author") or {}).get("role") or "").lower()
            if role == "user":
                continue
            found_files, found_sediments = cls._asset_ids_from_payload(
                message.get("content") or {}, excluded_file_ids,
            )
            for asset_id in found_files:
                if asset_id not in file_ids:
                    file_ids.append(asset_id)
            for asset_id in found_sediments:
                if asset_id not in sediment_ids:
                    sediment_ids.append(asset_id)
        return file_ids, sediment_ids

    def _conversation_image_asset_ids(
        self, conversation_id: str, excluded_file_ids: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """读取当前会话已落盘的生图结果。"""
        path = f"/backend-api/conversation/{conversation_id}"
        response = self.session.get(self.base_url + path, headers=self._headers(path), timeout=30)
        ensure_ok(response, path)
        return self._conversation_generated_image_asset_ids((response.json().get("mapping") or {}), excluded_file_ids)

    def _poll_image_asset_ids(self, conversation_id: str, timeout_secs: float = 120.0, expected_count: int = 1,
                              excluded_file_ids: set[str] | None = None, initial_file_ids: list[str] | None = None,
                              initial_sediment_ids: list[str] | None = None) -> tuple[list[str], list[str]]:
        """轮询会话，直到 GPT 的生图工具输出可下载资产。"""
        start = time.time()
        file_ids: list[str] = list(initial_file_ids or [])
        sediment_ids: list[str] = list(initial_sediment_ids or [])

        if len(file_ids) + len(sediment_ids) >= expected_count:
            return file_ids, sediment_ids

        # Initial wait for upstream async task
        time.sleep(4.0)

        while (time.time() - start) < timeout_secs:
            try:
                found_files, found_sediments = self._conversation_image_asset_ids(conversation_id, excluded_file_ids)
                for asset_id in found_files:
                    if asset_id not in file_ids:
                        file_ids.append(asset_id)
                for asset_id in found_sediments:
                    if asset_id not in sediment_ids:
                        sediment_ids.append(asset_id)
            except Exception as exc:
                logger.warning(f"Error polling GPT image result for conversation {conversation_id}: {exc}")
            if len(file_ids) + len(sediment_ids) >= expected_count:
                return file_ids, sediment_ids

            time.sleep(3.0)

        if not file_ids and not sediment_ids:
            raise TimeoutError(f"Image generation timed out after {timeout_secs}s")
        return file_ids, sediment_ids

    def _get_file_download_url(self, file_id: str) -> str:
        """Get CDN download URL for a file_id."""
        path = f"/backend-api/files/{file_id}/download"
        response = self.session.get(self.base_url + path, headers=self._headers(path, {"Accept": "application/json"}), timeout=60)
        ensure_ok(response, path)
        data = response.json()
        return data.get("download_url") or data.get("url") or ""

    def _get_attachment_download_url(self, conversation_id: str, attachment_id: str) -> str:
        """解析 GPT 会话附件类型的生图结果下载地址。"""
        path = f"/backend-api/conversation/{conversation_id}/attachment/{attachment_id}/download"
        response = self.session.get(self.base_url + path, headers=self._headers(path, {"Accept": "application/json"}), timeout=60)
        ensure_ok(response, path)
        data = response.json()
        return data.get("download_url") or data.get("url") or ""

    def delete_conversation_async(self, conversation_id: str) -> None:
        """Asynchronously delete temporary conversation to keep account clean."""
        if not conversation_id:
            return

        def _delete():
            try:
                path = f"/backend-api/conversation/{conversation_id}"
                self.session.patch(
                    self.base_url + path,
                    headers=self._headers(path, {"Content-Type": "application/json"}),
                    json={"is_visible": False},
                    timeout=15,
                )
            except Exception:
                pass

        threading.Thread(target=_delete, daemon=True).start()

    def generate_image_bytes(self, prompt: str, model: str = "gpt-image-2", references: list[str] | None = None, expected_count: int = 1) -> bytes | list[bytes]:
        """Main entry: zero-storage in-memory image generation."""
        if not self.access_token:
            raise ValueError("access_token is required for image generation")

        # 1. Warm up PoW resources (cached)
        self._bootstrap()

        # 2. Compute Sentinel Token and Proof Token
        reqs = self._get_chat_requirements()

        # 3. Prepare conversation, get conduit_token
        conduit_token = self._prepare_image_conversation(prompt, reqs, model)

        # 4. Start image generation long connection
        response, reference_hashes, reference_file_ids = self._start_image_generation(prompt, reqs, conduit_token, model, references)

        conversation_id = ""
        sse_file_ids: list[str] = []
        sse_sediment_ids: list[str] = []
        try:
            # /f/conversation 的 Patch 事件将生成图片写在 v.message，优先直接取图。
            for payload in iter_sse_payloads(response):
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                    if isinstance(event, dict):
                        quota_error = self._sse_image_quota_error(event)
                        if quota_error:
                            raise ImageQuotaExceededError(*quota_error)
                        event_conversation_id, found_files, found_sediments = self._sse_image_asset_ids(
                            event, reference_file_ids,
                        )
                        if event_conversation_id:
                            conversation_id = event_conversation_id
                        for asset_id in found_files:
                            if asset_id not in sse_file_ids:
                                sse_file_ids.append(asset_id)
                        for asset_id in found_sediments:
                            if asset_id not in sse_sediment_ids:
                                sse_sediment_ids.append(asset_id)
                        # 图片工具消息已经给出足量资源时立即下载，不能被后续文本 SSE 长连接阻塞。
                        if conversation_id and len(sse_file_ids) + len(sse_sediment_ids) >= expected_count:
                            break
                except ImageQuotaExceededError:
                    raise
                except Exception:
                    pass
        finally:
            response.close()

        if not conversation_id:
            raise RuntimeError("Upstream SSE did not return conversation_id")

        completed = False
        try:
            # 5. Poll for image file IDs
            file_ids, sediment_ids = self._poll_image_asset_ids(
                conversation_id,
                expected_count=expected_count,
                excluded_file_ids=reference_file_ids,
                initial_file_ids=sse_file_ids,
                initial_sediment_ids=sse_sediment_ids,
            )

            # 6. Download all results
            images: list[bytes] = []
            for source, asset_id in [("file", item) for item in file_ids] + [("attachment", item) for item in sediment_ids]:
                download_url = (
                    self._get_file_download_url(asset_id)
                    if source == "file"
                    else self._get_attachment_download_url(conversation_id, asset_id)
                )
                if not download_url:
                    raise RuntimeError(f"Could not resolve download url for {source} asset {asset_id}")
                img_resp = self.session.get(download_url, timeout=60)
                ensure_ok(img_resp, "download_image_bytes")
                if hashlib.sha256(img_resp.content).digest() in reference_hashes:
                    raise RuntimeError("Upstream returned an uploaded reference instead of a generated image")
                images.append(img_resp.content)
            completed = True
            return images[0] if len(images) == 1 else images
        finally:
            # 仅在图片已完整取回后清理会话，超时时保留网页端结果供后续人工恢复。
            if completed:
                self.delete_conversation_async(conversation_id)
