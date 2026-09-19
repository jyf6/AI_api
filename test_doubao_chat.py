import asyncio
import urllib.request
import json
from providers.doubao.backend import DoubaoBackendAPI

async def main():
    res = urllib.request.urlopen('http://127.0.0.1:8000/api/doubao/accounts')
    data = json.loads(res.read())
    cookies = data['accounts'][0]['cookies']
    async with DoubaoBackendAPI(cookies) as backend:
        try:
            res = await backend.chat("Hi")
            print("Chat response:", res)
        except Exception as e:
            print(f"Chat error: {e}")

asyncio.run(main())
