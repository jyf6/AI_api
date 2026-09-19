import asyncio
import urllib.request
import json
from providers.doubao.backend import DoubaoBackendAPI

async def main():
    res = urllib.request.urlopen('http://127.0.0.1:8000/api/doubao/accounts')
    data = json.loads(res.read())
    cookies = data['accounts'][0]['cookies']
    url = "http://localhost:9000/flexi-media/worksheet/WS20260908009/img_6ebc31a0-5527-4d10-b034-be5e4f5b40b6.png"
    
    try:
        async with DoubaoBackendAPI(cookies) as backend:
            res = await backend.upload_image(url)
            print(res)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
