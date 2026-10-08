import pytest
from pydantic import ValidationError

from api.schemas import ImageGenerationRequest
from utils.image_ratio import parse_and_normalize_ratio


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
