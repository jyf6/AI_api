from providers.gemini import backend


def test_oss_reference_is_read_with_sdk(monkeypatch):
    keys = []

    def read_object(key):
        keys.append(key)
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(backend, "read_oss_reference", read_object)

    content, extension = backend._read_image_source("tasks/reference image.png")

    assert keys == ["tasks/reference image.png"]
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    assert extension == ".png"
