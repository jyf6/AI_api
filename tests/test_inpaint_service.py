import io

from fastapi import UploadFile
from PIL import Image

from inpaint_service import app as inpaint_app


def png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_lama_result_only_replaces_masked_area(monkeypatch):
    original = Image.new("RGB", (32, 24), (20, 40, 60))
    mask = Image.new("L", original.size, 0)
    for x in range(10, 20):
        for y in range(8, 16):
            mask.putpixel((x, y), 255)

    class Model:
        def __call__(self, image, received_mask):
            assert image.size == original.size
            assert received_mask.getpixel((15, 12)) == 255
            return Image.new("RGB", (32, 24), (240, 20, 20))

    monkeypatch.setattr(inpaint_app, "model", Model())
    response = inpaint_app.inpaint(
        UploadFile(file=io.BytesIO(png(original))),
        UploadFile(file=io.BytesIO(png(mask))),
        radius=3,
    )
    result = Image.open(io.BytesIO(response.body))
    assert result.size == original.size
    assert result.getpixel((0, 0)) == original.getpixel((0, 0))
    assert result.getpixel((15, 12)) == (240, 20, 20)
