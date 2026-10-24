"""初版分析阶段的文本校验回归测试。"""

from unittest.mock import patch

with patch("core.database.database.list_accounts", return_value=[]):
    from api.routers.chat import is_anomalous_text


def test_long_reference_and_composition_outputs_are_allowed():
    long_text = "A detailed visual description. " * 180
    assert len(long_text) > 4000
    assert not is_anomalous_text(long_text, "INITIAL_ITEM_REF_ANALYSIS")
    assert not is_anomalous_text(long_text, "INITIAL_ITEM_COMPOSITION_PLANNING")
    assert not is_anomalous_text(long_text, "INITIAL_GLOBAL_ANALYSIS")


def test_short_english_product_selection_is_allowed_only_at_selection_stage():
    selection = "Selected product images: 1, 3"
    assert not is_anomalous_text(selection, "INITIAL_ITEM_PRODUCT_SELECTION")
    assert is_anomalous_text(selection, "INITIAL_ITEM_COMPOSITION_PLANNING")
    assert is_anomalous_text("Selected product images: none", "INITIAL_ITEM_PRODUCT_SELECTION")


def test_short_invalid_text_and_existing_json_exemption_remain():
    assert is_anomalous_text("Oops", "INITIAL_ITEM_REF_ANALYSIS")
    assert not is_anomalous_text('{"value": 1}', "INITIAL_GLOBAL_ANALYSIS")


def test_explicit_empty_product_selection_is_allowed_only_in_selection_stage():
    # 与 Java 单行协议一致：只有完整 None 是合法空选择，不能放过格式错误。
    selection = "Selected product images: None"
    assert not is_anomalous_text(selection, "INITIAL_ITEM_PRODUCT_SELECTION")
    assert is_anomalous_text(selection, "INITIAL_ITEM_COMPOSITION_PLANNING")
    for invalid in ("None", "Selected product images: none", "Selected product images: []",
                    "Selected product images: None, 1", "Selected product images: None because no product"):
        assert is_anomalous_text(invalid, "INITIAL_ITEM_PRODUCT_SELECTION")
