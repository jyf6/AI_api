from __future__ import annotations
import io

import pytest

from utils import oss_reference


def test_reference_uses_configured_bucket_and_object_key(monkeypatch):
    calls = []

    class Bucket:
        def get_object(self, key, *, process):
            calls.append(key)
            assert "resize" in process
            return io.BytesIO(b"\xff\xd8\xffimage")

        def read(self):
            return b"image"

    monkeypatch.setattr(oss_reference, "_bucket", lambda: Bucket())
    assert oss_reference.read_oss_reference("images/example.png") == b"\xff\xd8\xffimage"
    assert calls == ["images/example.png"]


def test_reference_rejects_url_before_oss_request(monkeypatch):
    monkeypatch.setattr(oss_reference, "_bucket", lambda: pytest.fail("unexpected OSS request"))
    with pytest.raises(ValueError, match="objectKey"):
        oss_reference.read_oss_reference("https://other.example/image.png")
