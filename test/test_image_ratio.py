import io
import pytest
from PIL import Image
from pydantic import ValidationError

from api.schemas import ImageGenerationRequest
from utils.image_ratio import normalize_image_ratio, parse_and_normalize_ratio


def test_ratio_parsing_and_normalization():
    # 预设比例
    assert parse_and_normalize_ratio("1:1") == (1, 1, "1:1")
    assert parse_and_normalize_ratio("9:16") == (9, 16, "9:16")
    assert parse_and_normalize_ratio("3:4") == (3, 4, "3:4")
    assert parse_and_normalize_ratio("4:3") == (4, 3, "4:3")
    assert parse_and_normalize_ratio("16:9") == (16, 9, "16:9")

    # 非约分值自动约分
    assert parse_and_normalize_ratio("12:16") == (3, 4, "3:4")
    assert parse_and_normalize_ratio("60:80") == (3, 4, "3:4")
    assert parse_and_normalize_ratio("10:10") == (1, 1, "1:1")
    assert parse_and_normalize_ratio("5:7") == (5, 7, "5:7")


def test_ratio_out_of_bounds_rejected():
    # 边界 1:4 至 4:1
    assert parse_and_normalize_ratio("1:4") == (1, 4, "1:4")
    assert parse_and_normalize_ratio("4:1") == (4, 1, "4:1")

    # 超出边界
    with pytest.raises(ValueError, match="超出 1:4 至 4:1 范围"):
        parse_and_normalize_ratio("1:5")

    with pytest.raises(ValueError, match="超出 1:4 至 4:1 范围"):
        parse_and_normalize_ratio("5:1")

    # 宽高超出 1..100
    with pytest.raises(ValueError, match="宽高值必须在 1 至 100 之间"):
        parse_and_normalize_ratio("101:1")


def test_image_generation_request_schema():
    # 正常请求
    req = ImageGenerationRequest(prompt="test", aspect_ratio="3:4")
    assert req.aspect_ratio == "3:4"

    # 非约分值在 schema 层面自动规范化
    req2 = ImageGenerationRequest(prompt="test", aspect_ratio="12:16")
    assert req2.aspect_ratio == "3:4"

    # 越界比例抛出验证错误
    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="test", aspect_ratio="1:5")


def test_normalize_image_ratio_padding_when_mismatched():
    # 创建一张 1000x1000 (1:1) 的红色正方形测试图
    src_img = Image.new("RGB", (1000, 1000), (255, 0, 0))
    out = io.BytesIO()
    src_img.save(out, format="PNG")
    raw_bytes = out.getvalue()

    # 目标比例为 3:4
    padded_bytes = normalize_image_ratio(raw_bytes, "3:4")
    assert padded_bytes is not None

    with Image.open(io.BytesIO(padded_bytes)) as result_img:
        w, h = result_img.size
        # 断言长边保持原图长边 1000，高为 1000，宽为 750 (1000 * 3 / 4)
        assert h == 1000
        assert w == 750
        assert w * 4 == h * 3  # 严格 3:4

        # 检查背景色：上部 (375, 50) 和 下部 (375, 950) 为白色画布补边
        # 居中内容区域 (375, 500) 为红色
        assert result_img.getpixel((375, 50)) == (255, 255, 255)
        assert result_img.getpixel((375, 950)) == (255, 255, 255)
        assert result_img.getpixel((375, 500)) == (255, 0, 0)


def test_normalize_image_ratio_landscape_padding():
    # 原图 800x800 正方形，目标 16:9
    src_img = Image.new("RGB", (800, 800), (0, 255, 0))
    out = io.BytesIO()
    src_img.save(out, format="PNG")
    raw_bytes = out.getvalue()

    padded_bytes = normalize_image_ratio(raw_bytes, "16:9")
    with Image.open(io.BytesIO(padded_bytes)) as result_img:
        w, h = result_img.size
        # 长边保持原图 800，高为 800 * 9 / 16 = 450
        assert w == 800
        assert h == 450
        assert w * 9 == h * 16  # 严格 16:9
        # 左侧补白，中间为绿色
        assert result_img.getpixel((20, 225)) == (255, 255, 255)
        assert result_img.getpixel((400, 225)) == (0, 255, 0)
