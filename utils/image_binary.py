from __future__ import annotations


def image_media_type(data: bytes | bytearray) -> str | None:
    """根据图片魔数确认二进制图片类型，拒绝被伪装成图片的错误响应。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None
