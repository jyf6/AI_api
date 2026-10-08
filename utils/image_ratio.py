from __future__ import annotations

import math


def parse_and_normalize_ratio(aspect_ratio: str) -> tuple[int, int, str]:
    """校验并约分宽高比。
    格式必须为 W:H，宽高均为 1~100 的整数，整体介于 1:4 至 4:1 之间。
    非约分值自动规范化（如 1200:1600 规范化为 3:4）。
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

    if w < 1 or w > 100 or h < 1 or h > 100:
        raise ValueError(f"aspect_ratio 宽高值必须在 1 至 100 之间，收到: {w}:{h}")

    # 最大公约数约分
    g = math.gcd(w, h)
    w //= g
    h //= g

    ratio_val = w / h
    if ratio_val < 0.25 or ratio_val > 4.0:
        raise ValueError(f"aspect_ratio 超出 1:4 至 4:1 范围，约分后为: {w}:{h}")

    return w, h, f"{w}:{h}"


def build_ratio_prompt(prompt: str, aspect_ratio: str) -> str:
    """在提示词中显式加入画幅要求，作为各平台原生比例控制之外的双重保险。

    明确告知模型：即使提供了参考图（垫图），也不得沿用其比例，
    必须严格按目标宽高比生成。aspect_ratio 会被再次校验并约分。
    """
    _, _, ratio = parse_and_normalize_ratio(aspect_ratio)
    instruction = (
        f"\n\n【画幅要求】必须严格按照宽高比 {ratio} 生成图片。"
        f"即使本次提供了参考图（垫图），也绝对不要沿用参考图的比例，"
        f"最终输出图片的宽高比必须精确等于 {ratio}，不得因垫图而改变。"
    )
    return f"{prompt}{instruction}"
