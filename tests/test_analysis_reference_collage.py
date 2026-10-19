"""分析拼图的分组、重试和请求资源生命周期验证。"""
import asyncio
import io
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from pydantic import ValidationError

with patch('core.database.database.list_accounts', return_value=[]):
    from api.routers import chat
from api.schemas import ChatCompletionRequest
from utils import oss_reference
from utils.image_stitch import stitch_reference_sheets


def image_bytes():
    out = io.BytesIO()
    with Image.new('RGBA', (1200, 300), (255, 0, 0, 128)) as image:
        image.save(out, format='PNG')
    return out.getvalue()


@pytest.mark.parametrize('count,expected', [(2, [(1536, 768)]), (3, [(1536, 1536)]),
                                            (4, [(1536, 1536)]), (5, [(1536, 1536), (768, 768)])])
def test_reference_sheet_layout(count, expected):
    sheets = stitch_reference_sheets([image_bytes()] * count)
    sizes = []
    for sheet in sheets:
        assert len(sheet) < 3_000_000
        with Image.open(io.BytesIO(sheet)) as image:
            sizes.append(image.size)
            assert image.getpixel((100, 100)) == (255, 255, 255)
            assert image.crop((12, 4, 80, 36)).getextrema()[0][0] < 50
    assert sizes == expected


def test_count_cannot_exceed_images():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(prompt='p', images=['a'], reference_image_count=2)


@pytest.mark.parametrize('count', [None, 0, 1, 2, 5])
def test_references_passed_unchanged(monkeypatch, count):
    """验证多模态分析接口不再进行代理侧拼图，所有参考图均以原图形式直接透传给模型。"""
    images = ['reference_1', 'reference_2', 'reference_3', 'reference_4', 'reference_5']
    body = ChatCompletionRequest(prompt='p', images=images, reference_image_count=count)
    async def prepared(actual):
        assert actual.images == body.images
        return 'ok'
    monkeypatch.setattr(chat, '_chat_completions_prepared', prepared)
    assert asyncio.run(chat._chat_completions(body)) == 'ok'

