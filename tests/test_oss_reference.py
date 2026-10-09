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
        oss_reference.read_oss_reference("file:///image.png")


@pytest.mark.parametrize("url", ["https://static-worksheet.nantang-tech.com/a.jpg?version=1",
                                 "https://nt-worksheet.oss-cn-shanghai.aliyuncs.com/a.jpg",
                                 "https://m.media-amazon.com/images/a.jpg", "https://images.example.org:8080/a.jpg"])
def test_url_reference_is_cached_without_touching_current_bucket(monkeypatch, url):
    calls = []
    monkeypatch.setattr(oss_reference, "_bucket", lambda: pytest.fail("wrong bucket"))
    def fetch(source, dimension):
        calls.append((source, dimension))
        return b"\xff\xd8\xffimage"
    monkeypatch.setattr(oss_reference, "_fetch_url_reference", fetch)
    assert oss_reference.read_oss_reference(url, 480) == b"\xff\xd8\xffimage"
    assert oss_reference.read_oss_reference(url, 480) == b"\xff\xd8\xffimage"
    assert calls == [(url, 480)]


@pytest.mark.parametrize("url", ["file:///image.png", "ftp://example.org/a.jpg", "https://user:pass@m.media-amazon.com/a.jpg"])
def test_untrusted_url_is_rejected_before_download(monkeypatch, url):
    monkeypatch.setattr(oss_reference, "build_opener", lambda *_: pytest.fail("unexpected download"))
    with pytest.raises(ValueError):
        oss_reference.read_oss_reference(url)


def test_redirect_cannot_escape_reference_hosts():
    with pytest.raises(ValueError):
        oss_reference._ReferenceRedirectHandler().redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/private")


def test_url_is_decoded_resized_and_bounded(monkeypatch):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (1000, 600), "red").save(buffer, "PNG")
    class Opener:
        def open(self, request, timeout):
            assert timeout == 15
            return io.BytesIO(buffer.getvalue())
    monkeypatch.setattr(oss_reference, "_check_reference_address", lambda _: None)
    monkeypatch.setattr(oss_reference, "build_opener", lambda *_: Opener())
    data = oss_reference._fetch_url_reference("https://m.media-amazon.com/images/a.png", 480)
    assert len(data) < 3_000_000
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (480, 288)


def test_private_dns_resolution_is_rejected_before_download(monkeypatch):
    monkeypatch.setattr(oss_reference.socket, "getaddrinfo", lambda *_args, **_kw: [(2, 1, 6, "", ("192.168.1.10", 443))])
    monkeypatch.setattr(oss_reference, "build_opener", lambda *_: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="private"):
        oss_reference._fetch_url_reference("https://images.example.org/a.jpg", 480)
