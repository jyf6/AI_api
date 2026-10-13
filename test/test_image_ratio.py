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


def test_custom_ratio_without_artificial_bounds():
    assert parse_and_normalize_ratio("1:5") == (1, 5, "1:5")
    assert parse_and_normalize_ratio("101:1") == (101, 1, "101:1")
    assert parse_and_normalize_ratio("1200:800") == (3, 2, "3:2")
    with pytest.raises(ValueError, match="正整数"):
        parse_and_normalize_ratio("0:5")


def test_image_generation_request_schema():
    # 正常请求
    req = ImageGenerationRequest(prompt="test", aspect_ratio="3:4")
    assert req.aspect_ratio == "3:4"

    # 非约分值在 schema 层面自动规范化
    req2 = ImageGenerationRequest(prompt="test", aspect_ratio="12:16")
    assert req2.aspect_ratio == "3:4"

    assert ImageGenerationRequest(prompt="test", aspect_ratio="1:5").aspect_ratio == "1:5"
    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="test", aspect_ratio="0:5")
