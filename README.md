# ChatGPT 轻量生图微服务 (ChatGPT Image Service)

基于网页端逆向调用的轻量模型微服务，支持 ChatGPT、豆包和 Gemini 的文本、多模态分析与生图；生成图片以内存二进制返回，不保存业务图片。

---

## 🌟 核心特性

- **轻量服务**：提供生图与文本、多模态接口，账号数据保存在 MySQL。
- **账号独立代理 (可选)**：支持为每个账号绑定独立的 HTTP/SOCKS5 代理节点，严格实现“一号一节点”网络隔离，防关联封号；未配置则自动宿主直连。
- **单一稳定登录**：仅支持 OAuth Refresh Token 模式，通过账号专属代理自动刷新 `access_token` 并保活。
- **Zero-Storage 纯内存直出**：生成的图片以原图二进制返回，本地不产生业务图片文件。
- **槽位会话续用**：有工单与槽位标识时，GPT/Gemini 优先续用该平台当前账号和会话，状态缓存在既有 Redis 中；GPT 成功后保留会话，豆包保持直接调用。
- **内置调试面板**：根路径 `GET /` 内置原生免构建单页 Web UI，支持实时查看号池、录号、在线生图与图片预览。
- **Gemini Web 逆向调用**：粘贴 Gemini Cookie（含 `__Secure-1PSID`）即可调用文本、多模态和生图；不启动浏览器自动化。
- **微服务无缝接入**：对外提供标准 OpenAI `/v1/images/generations` 接口，可直接被上游业务微服务无缝调用。

---

## 🚀 快速开始

### Docker Compose 启动

```bash
cp .env.example .env
# 编辑 .env，填写与 Java 服务相同的 OSS_ACCESS_ID、OSS_ACCESS_KEY、OSS_ENDPOINT、OSS_BUCKET
# MySQL、Redis 使用 flexi-scheduler-shared 网络中的 mysql、redis 服务
docker compose up -d --build app
```

Docker 部署使用 `docker-compose.yml`；先启动共享网络中的 MySQL、Redis，并由 Java Flyway 建表。Python 从环境变量读取数据库、Redis 和 OSS 配置；参考图参数为当前 OSS bucket 的 objectKey。抠图模型缓存保存在 Docker 卷中，首次调用抠图接口时才加载。服务默认只监听宿主机 `127.0.0.1:8010`。

启动后可访问 `http://localhost:8010/`；生图接口为 `POST /v1/images/generations`，健康检查为 `GET /health`。

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

### AI 套图业务的 GPT 模型策略

- 分析接口传 `model: "gpt-chat"`，上游固定使用 `gpt-5-6-thinking` + `thinking_effort: "extended"`。
- 生图及图片调整传 `model: "gpt-image"`，上游固定使用 `gpt-6-thinking` + `thinking_effort: "extended"`；会话准备和正式提交的模型、思考参数一致。
- 模型不可用时沿用现有失败及重试流程，不自动退回 `auto`。账号健康检查仍使用 `auto`，不附加思考参数。
- 以上是请求策略，账号是否支持 5.6 扩展思考及 6 扩展思考生图需通过实际调用确认；参数测试不代表真实模型已经执行成功。

### 3. Gemini Cookie 配置与模型

GPT、Gemini、豆包的分析及生图，每次模型调用统一最多等待 **240 秒（4 分钟）**。提交、流式接收、轮询和图片取回共用预算；重试按新的一次尝试计时。Gemini 分析不再因正文停顿 10 秒而提前返回部分结果。Java 外层 300 秒用于容纳准入排队和素材准备，不代表模型单次等待 5 分钟。

在控制台的“Gemini Web Cookie 账号”中粘贴 `gemini.google.com` Cookie（必须含 `__Secure-1PSID`，建议同时包含 `__Secure-1PSIDTS` 和完整 Cookie）。Gemini 账号独立保存为 `data/gemini_accounts.json`。

- 文本/多模态分析：`POST /v1/chat/completions`，传 `model: "gemini-chat"`，统一使用账号清单中的 **3.8 Flash + 高思考（扩展思考）**，当前只支持 `stream: false`。
- 生图及图片调整：`POST /v1/images/generations`，传 `model: "gemini-image"`，统一使用账号清单中的 **Pro + 高思考（扩展思考）**。
- 模型名称/别名精确匹配后传入真实模型对象；不硬编码内部 ID，不将缺失的 3.8 Flash 替换成其他版本或默认模型。账号缺少目标模型或模型不可用时，请求明确失败。日志中的 `gemini_model_selected` 记录选定的名称、ID 和扩展思考开关，不记录 Cookie。
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
    base_url="http://localhost:8010/v1",
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
