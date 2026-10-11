import math
import io
from PIL import Image, ImageDraw, ImageFont

def _ensure_rgb(img):
    if img.mode in ("RGBA", "P", "LA"):
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        if img.mode == "RGBA" or img.mode == "LA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            mask = img.split()[-1]
            background.paste(img, mask=mask)
            return background
        return img.convert("RGB")
    return img

def stitch_images_to_bytes(image_bytes_list: list[bytes], max_width=1024) -> bytes:
    if not image_bytes_list:
        raise ValueError("No images to stitch")
        
    images = []
    for b in image_bytes_list:
        # 任一素材失败都停止，不允许缺图拼接。
        with Image.open(io.BytesIO(b)) as source:
            source.load()
            images.append(_ensure_rgb(source).copy())

    if not images:
        raise ValueError("Failed to open any images for stitching")

    count = len(images)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    cell_size = 512
    for i in range(len(images)):
        img = images[i]
        img.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    
    grid_image = Image.new('RGB', (grid_w, grid_h), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_size + (cell_size - img.width) // 2
        y = row * cell_size + (cell_size - img.height) // 2
        grid_image.paste(img, (x, y))
        
    if grid_image.width > max_width:
        ratio = max_width / grid_image.width
        new_h = int(grid_image.height * ratio)
        grid_image = grid_image.resize((max_width, new_h), Image.Resampling.LANCZOS)
        
    out = io.BytesIO()
    return _encode_bounded_jpeg(grid_image)

def _encode_bounded_jpeg(image: Image.Image) -> bytes:
    """与 Java 同步：先降质量，再缩画布，最终严格小于 3MB。"""
    quality = 85
    current = image
    for _ in range(16):
        out = io.BytesIO()
        current.save(out, format="JPEG", quality=quality, optimize=True)
        data = out.getvalue()
        if len(data) < 3_000_000:
            return data
        if quality > 65:
            quality = max(65, quality - 10)
        else:
            width, height = int(current.width * .8), int(current.height * .8)
            if min(width, height) < 256:
                break
            current = current.resize((width, height), Image.Resampling.LANCZOS)
    raise ValueError("Contact sheet cannot be reduced below 3 MB")


def stitch_reference_sheets(image_bytes_list: list[bytes]) -> list[bytes]:
    """参考图每四张一页，连续 R 编号与图片内容分区，保持原始顺序。"""
    sheets = []
    font = ImageFont.load_default(size=28)
    for start in range(0, len(image_bytes_list), 4):
        group = image_bytes_list[start:start + 4]
        cols = min(2, len(group))
        rows = math.ceil(len(group) / cols)
        # 768 像素单元含 40 像素编号区，整页最长边不超过 1536。
        with Image.new("RGB", (cols * 768, rows * 768), "white") as canvas:
            draw = ImageDraw.Draw(canvas)
            for offset, data in enumerate(group):
                x, y = (offset % cols) * 768, (offset // cols) * 768
                draw.text((x + 12, y + 4), f"R{start + offset + 1}", fill="black", font=font)
                with Image.open(io.BytesIO(data)) as source:
                    rgb = _ensure_rgb(source).convert("RGB")
                    try:
                        rgb.thumbnail((768, 728), Image.Resampling.LANCZOS)
                        canvas.paste(rgb, (x + (768 - rgb.width) // 2, y + 40 + (728 - rgb.height) // 2))
                    finally:
                        rgb.close()
            sheets.append(_encode_bounded_jpeg(canvas))
    return sheets
