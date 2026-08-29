from __future__ import annotations

import base64
import io
from typing import Any

from curl_cffi.requests import AsyncSession


class GeminiBackendAPI:
    """Gemini Web reverse client; no browser process and no persistent image files."""

    def __init__(self, account: dict[str, Any]) -> None:
        self.account = account
        self.client = None

    async def __aenter__(self):
        from gemini_webapi import GeminiClient
        self.client = GeminiClient(self.account["psid"], self.account.get("psidts") or None,
                                   proxy=self.account.get("proxy") or None)
        if self.account.get("cookie"):
            self.client.cookies.update(_cookie_dict(self.account["cookie"]))
        await self.client.init(timeout=180, auto_refresh=True, impersonate="chrome145")
        return self

    async def __aexit__(self, *_args):
        if self.client:
            from services.gemini_account_service import gemini_account_service
            gemini_account_service.merge_cookie(self.account["name"], dict(self.client.cookies))
            await self.client.close()
        self.client = None

    async def chat(self, messages: list[dict[str, Any]], model: str) -> str:
        prompt, files = _messages_to_prompt_and_files(messages)
        output = await self.client.generate_content(prompt, files=files or None, model=_runtime_model(model))
        text = (output.candidates[output.chosen].text if output.candidates else "")
        if not text:
            text = getattr(output, "text", "") or ""
        if not text.strip():
            raise RuntimeError("Gemini returned no assistant text")
        return text.strip()

    async def image(self, prompt: str, model: str, references: list[str] | None = None) -> list[bytes]:
        files = [_data_url_to_bytes(value) for value in (references or [])]
        output = await self.client.generate_content(prompt, files=files or None, model=_runtime_model(model))
        images = list(output.candidates[output.chosen].generated_images if output.candidates else [])
        if not images:
            raise RuntimeError((getattr(output, "text", "") or "Gemini returned no image").strip())
        return [await _download_generated_image(image, self.client, self.account.get("proxy") or None)
                for image in images]


def _cookie_dict(raw: str) -> dict[str, str]:
    return {part.split("=", 1)[0].strip(): part.split("=", 1)[1].strip()
            for part in raw.split(";") if "=" in part}


def _runtime_model(model: str) -> str:
    """Map UI/API aliases to names accepted by the installed Gemini Web client."""
    aliases = {
        "gemini-2.5-pro-image": "gemini-pro",
        "gemini-2.5-flash-image": "gemini-flash",
    }
    return aliases.get(model, model)


def _data_url_to_bytes(value: str) -> io.BytesIO:
    if not value.startswith("data:") or "," not in value:
        raise ValueError("Gemini currently accepts image data URLs only")
    payload = value.split(",", 1)[1]
    return io.BytesIO(base64.b64decode(payload))


def _messages_to_prompt_and_files(messages: list[dict[str, Any]]) -> tuple[str, list[io.BytesIO]]:
    texts: list[str] = []
    files: list[io.BytesIO] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif part.get("type") in {"image_url", "input_image"}:
                source = part.get("image_url") or part.get("url")
                source = source.get("url") if isinstance(source, dict) else source
                if source:
                    files.append(_data_url_to_bytes(str(source)))
    return "\n\n".join(texts), files


async def _download_generated_image(image: Any, client: Any, proxy: str | None) -> bytes:
    """Download into memory; generated images are never persisted by this service."""
    url = getattr(image, "url", "")
    if all(getattr(image, key, "") for key in ("cid", "rid", "rcid", "image_id")):
        full_size_url = await client._get_full_size_image(cid=image.cid, rid=image.rid, rcid=image.rcid,
                                                           image_id=image.image_id)
        if full_size_url:
            url = full_size_url + "=d-I?alr=yes"
    if not url:
        raise RuntimeError("Gemini returned an image without a download URL")
    async with AsyncSession(impersonate="chrome145", cookies=client.cookies, proxy=proxy) as session:
        response = await session.get(url, headers={"Referer": "https://gemini.google.com/"})
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("text/"):
            response = await session.get(response.text, headers={"Referer": "https://gemini.google.com/"})
            response.raise_for_status()
        return response.content
