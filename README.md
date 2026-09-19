# ChatGPT 轻量生图微服务 (ChatGPT Image Service)

基于网页端逆向调用的轻量模型微服务，支持 ChatGPT、豆包和 Gemini 的文本、多模态分析与生图；图片始终以内存 Base64 形式返回，不保存业务图片。

---

## 🌟 核心特性

- **专注生图**：剔除所有文本对话、Codex、沙箱 PPT/PSD、数据库等冗余功能，极低内存与 CPU 占用。
- **账号独立代理 (可选)**：支持为每个账号绑定独立的 HTTP/SOCKS5 代理节点，严格实现“一号一节点”网络隔离，防关联封号；未配置则自动宿主直连。
- **单一稳定登录**：仅支持 OAuth Refresh Token 模式，通过账号专属代理自动刷新 `access_token` 并保活。
- **Zero-Storage 纯内存直出**：生成的图片在内存中即生成即转 Base64 返回，本地不产生文件碎片，不落盘。
- **自动清理会话**：生图完成后自动异步清理 ChatGPT 官方后台临时对话，保持账号列表整洁。
- **内置调试面板**：根路径 `GET /` 内置原生免构建单页 Web UI，支持实时查看号池、录号、在线生图与图片预览。
- **Gemini Web 逆向调用**：粘贴 Gemini Cookie（含 `__Secure-1PSID`）即可调用文本、多模态和生图；不启动浏览器自动化。
- **微服务无缝接入**：对外提供标准 OpenAI `/v1/images/generations` 接口，可直接被上游业务微服务无缝调用。

---

## 🚀 快速开始

### 1. 本地直接运行 (推荐使用 uv)

```bash
# 安装依赖
uv sync

# 启动微服务 (默认端口 8000)
uv run python main.py
```

启动后访问：
- **Web 调试与号池面板**：`http://localhost:8000/`
- **生图 API 接口**：`POST http://localhost:8000/v1/images/generations`
- **健康检查**：`GET http://localhost:8000/health`

### 2. Docker 容器化运行

```bash
docker build -t chatgpt-image-service .
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data --name img-service chatgpt-image-service
```

---

## 🔌 API 接口规范

### 1. 核心文生图接口

- **路径**：`POST /v1/images/generations`
- **请求格式**：`application/json`

**请求示例**：
```json
{
  "prompt": "A cute cyberpunk cat coding on a laptop, neon lights, 4k",
  "model": "gpt-image-2",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json"
}
```

套图数量和风格完全由 `prompt` 决定，服务不会改写提示词；一次上游会话实际返回多少张就返回多少张。`response_format` 支持 `b64_json` 和 `url`。由于服务不保存图片，`url` 返回内存 `data:image/png;base64,...` 地址，仅适合内部调用链即时消费。

### 2. 网页 ChatGPT 多模态分析接口

- **路径**：`POST /v1/chat/completions`
- **用途**：调用网页 ChatGPT 普通聊天模型进行文本/图片分析；结果不会自动串联到生图接口。
- **模型**：默认 `gpt-5-5`，也可传入当前账号可用的网页模型名。
- **图片输入**：使用 OpenAI 风格 `image_url`，建议传 `data:image/...;base64,...`。

```json
{
  "model": "gpt-5-5",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "分析这张参考图的角色、风格和构图"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

### 3. Gemini Cookie 配置与模型

在控制台的“Gemini Web Cookie 账号”中粘贴 `gemini.google.com` Cookie（必须含 `__Secure-1PSID`，建议同时包含 `__Secure-1PSIDTS` 和完整 Cookie）。Gemini 账号独立保存为 `data/gemini_accounts.json`。

- 文本/多模态：`POST /v1/chat/completions`，模型如 `gemini-3-flash`，当前只支持 `stream: false`。
- 生图：`POST /v1/images/generations`，模型如 `gemini-2.5-pro-image`。
- 失败时本次请求直接返回失败并降低所用 Gemini 账号状态，不自动切换其他账号。

Gemini 通过项目内置的网页内部 HTTP 协议实现调用，不依赖 Playwright 或 Chromium。该实现保留了原始 GPL-3.0 许可证；使用或分发本服务前请审查其许可证条款。

**响应示例**：
```json
{
  "id": "chatcmpl-1756201200",
  "object": "chat.completion",
  "model": "gpt-5-5",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "分析结果..."},
    "finish_reason": "stop"
  }]
}
```

---

### 2. 号池管理接口 (RESTful)

| 方法 | 路径 | 功能说明 | 请求 Body 示例 |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/accounts` | 获取所有账号状态列表 | 无 |
| `POST` | `/api/accounts` | 录入/更新账号 (通过代理验证) | `{"email": "user@test.com", "refresh_token": "rt-xxx", "proxy": "http://ip:port"}` |
| `DELETE` | `/api/accounts/{email}` | 删除指定账号 | 无 |
| `POST` | `/api/accounts/{email}/refresh` | 手动强制刷新指定账号 Token | 无 |
| `GET` | `/api/stats` | 查询号池统计 (总数/可用数/在途) | 无 |
| `GET` | `/health` | 服务健康探活 | 无 |

---

## 💻 上游微服务调用示例 (Python)

使用官方 `openai` SDK 即可直接调用：

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

response = client.images.generate(
    model="gpt-image-2",
    prompt="A futuristic city floating in the sky, concept art",
    response_format="b64_json"
)

# 拿到图片的 Base64 字符串
image_base64 = response.data[0].b64_json
```
