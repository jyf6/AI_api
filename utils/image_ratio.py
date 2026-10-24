from __future__ import annotations


def parse_and_normalize_ratio(aspect_ratio: str) -> tuple[int, int, str]:
    """校验宽高比，不做约分。
    格式必须为 W:H，宽高均为正整数；不限制比例范围。
    原始宽高值原样保留（如 1200:1600 保持 1200:1600）。
    """
    if not aspect_ratio or not isinstance(aspect_ratio, str):
        raise ValueError("aspect_ratio 不能为空")

    parts = aspect_ratio.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"aspect_ratio 格式错误，必须为 W:H，收到: {aspect_ratio}")

    try:
        w, h = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise ValueError(f"aspect_ratio 必须为整数比例，收到: {aspect_ratio}") from exc

    if w < 1 or h < 1:
        raise ValueError(f"aspect_ratio 宽高值必须为正整数，收到: {w}:{h}")

    return w, h, f"{w}:{h}"


def build_ratio_prompt(prompt: str, aspect_ratio: str) -> str:
    """在提示词中说明生成图片的比例。aspect_ratio 只会被校验，不做约分。"""
    _, _, ratio = parse_and_normalize_ratio(aspect_ratio)
    return f"{prompt}\n\n生成图片的比例为 {ratio}。"
