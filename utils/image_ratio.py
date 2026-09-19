from __future__ import annotations

import io
import math
from PIL import Image


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


def normalize_image_ratio(image_bytes: bytes, target_ratio: str) -> bytes:
    """验证实际图片像素比例。若不一致，使用 Pillow 将原图等比缩放后居中放入目标比例的白色画布：
    不裁剪、不拉伸。长边保持原图长边像素，另一边向上补到偶数。最终统一输出 PNG。
    若比例已一致，仍统一输出 PNG 格式。
    """
    target_w, target_h, _ = parse_and_normalize_ratio(target_ratio)

    with Image.open(io.BytesIO(image_bytes)) as orig_img:
        orig_w, orig_h = orig_img.size

        # 判断原图比例是否已严格匹配目标比例
        if orig_w * target_h == orig_h * target_w and orig_img.format == "PNG":
            return image_bytes

        # 计算目标画布尺寸：长边保持原图长边像素，另一边向上补到偶数
        long_side = max(orig_w, orig_h)
        if target_w >= target_h:
            canvas_w = long_side
            canvas_h = round(long_side * target_h / target_w)
            if canvas_h % 2 != 0:
                canvas_h += 1
        else:
            canvas_h = long_side
            canvas_w = round(long_side * target_w / target_h)
            if canvas_w % 2 != 0:
                canvas_w += 1

        # 若原图宽高比与计算后的画布完全一致，且没有补边需求
        if orig_w == canvas_w and orig_h == canvas_h:
            out = io.BytesIO()
            orig_img.convert("RGB").save(out, format="PNG")
            return out.getvalue()

        # 等比缩放，不裁剪不拉伸
        scale = min(canvas_w / orig_w, canvas_h / orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))

        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = orig_img.resize((new_w, new_h), resample=resample)

        # 居中放置在白色画布
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2

        if resized.mode in ("RGBA", "LA") or (resized.mode == "P" and "transparency" in resized.info):
            rgba = resized.convert("RGBA")
            canvas.paste(rgba, (offset_x, offset_y), mask=rgba)
        else:
            canvas.paste(resized.convert("RGB"), (offset_x, offset_y))

        out = io.BytesIO()
        canvas.save(out, format="PNG")
        return out.getvalue()
