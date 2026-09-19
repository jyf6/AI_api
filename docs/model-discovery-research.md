# 网页逆向代理的模型发现与配置调研

调研日期：2026-09-18。范围是 ChatGPT / Gemini 网页端逆向客户端及统一网关的公开上游源码；它们不是官方 API，端点和返回结构都可能随网页端改变。

## 结论

“配置中允许填写模型名”不应被当成“账号必然能调用该模型”。可靠的模型配置应以**账号维度的上游发现结果**为准：保存每个账号、每种能力和每个模型的可用性与探测时间；请求先按模型筛选账号，再进行负载调度。全局模型列表只应是这些账号能力的交集（保证必定可用）或并集（必须标注哪些账号可用），而不是静态名单加上某一账号的结果。

## 上游项目的做法

| 项目 | 上游模型来源 | 选择/暴露策略 | 对本项目的启示 |
| --- | --- | --- | --- |
| [ChatGPT Web2API](https://github.com/Octo-Lex/ChatGPT-Web2API/blob/master/docs/protocol-reference.md) | 已捕获的 `GET /backend-api/models`，文档称其为完整模型目录。 | 证明 ChatGPT 网页账号应以该账号会话返回的 `slug` 与元数据作为真实网页模型事实。 | 对每个 GPT 账号用其自己的 access token 与 workspace/account header 拉取，不能把 API 模型 ID 当网页 slug。 |
| [lanqian528/chat2api](https://github.com/lanqian528/chat2api/blob/main/chatgpt/ChatService.py) | 调用方的 `model` 由 `model_proxy` 映射为网页请求模型。 | 以兼容别名/静态映射为主，多 Token 轮询与失败切号；README 说明支持多账号轮询和 token 刷新。 | 静态别名可保留为兼容层，但不是授权或可用性判断依据。 |
| [yukkcat/chatgpt2api](https://github.com/yukkcat/chatgpt2api/blob/main/services/model_catalog_service.py) | 文本模型优先取配置，否则回退到小型默认列表；图片模型可根据可用账号的类型推导。 | 生成统一、带 `source` 和 `revision` 的模型目录；其 [3.2.3 变更记录](https://github.com/yukkcat/chatgpt2api/blob/main/CHANGELOG.md) 明确说不再把上游实时发现的模型追加到公共目录。 | “稳定公共目录”与“逐账号事实”应分开：前者服务客户端兼容，后者服务调度，避免 UI 列表随上游波动。 |
| [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/blob/main/src/gemini_webapi/client.py) | `GeminiClient.init()` 调用用户状态 RPC，动态解析并建立 `_model_registry`；每项含网页端所需 header、容量和可用性。 | `list_models()` 返回当前账号注册表；`resolve_model()` 依序精确匹配 ID/名称/显示名、别名和规范化名称；找不到即报错。README 还明确建议不要硬编码名称。 | Gemini 已有正确的逐账号发现基础。应将注册表持久化/投影到账号能力表，并在调度前按该表筛选。 |
| [Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) | 使用 Gemini Web API，同时允许管理员在 YAML 中额外定义尚未被发现的模型 header。 | 对自定义条目原样透传 header，并在读模型/请求时重新加载配置。 | 将“已发现模型”和“管理员实验性覆盖”分层；覆盖项必须标记为未验证，不能自动视为所有账号可用。 |

## 各平台可获得的真实名称

### GPT / ChatGPT 网页账号

真实候选来自该账号调用 `GET /backend-api/models` 的结果。应至少保存返回对象的 `slug`、`title`、`description`、`capabilities`、`product_features` 与类别/订阅信息；请求时传的是网页端接受的 `slug`。这与公开 API 的模型 ID 属于不同模型面，因此不能把类似 `gpt-5.6-sol` 的 API 名称自动改写成网页端的 `gpt-5-6`。

如果账号可选择不同 ChatGPT workspace，发现与调用都必须使用相同的 `chatgpt-account-id`；否则发现到的是个人空间，而实际请求可能发往团队空间。

### Gemini 网页账号

上游库在初始化阶段的用户状态 RPC 中构造 `AvailableModel`，并缓存为 `_model_registry`；`list_models()` 返回当前账号的列表，`resolve_model()` 只在该列表中解析。其 README 指出模型名称、请求 header 和容量均由当前账号动态发现，且旧的硬编码枚举已弃用。模型记录至少应保存 `model_id`、`model_name`、`display_name`、`aliases`、`is_available`/不可用原因、模型 header、配额/容量及发现时间。

Gemini 的图片能力不宜只凭名称猜测。应把“能列出该模型”与“已完成小型生图探测”分为两个字段；后者可能消耗额度，应仅在显式测试或低频后台巡检时执行。

### 豆包网页账号

当前接入的网页请求不传独立模型字段，因此不存在可被当前协议可靠枚举、再按名字调度的模型集合。应在 UI 中明确展示为“网页默认能力”，不要提供看似可切换的自由模型名输入框；若未来抓到账号级模型选择端点，再按与 GPT/Gemini 相同的能力表接入。

## 推荐的数据与调度设计

新增的概念应是账号能力快照，而不是把模型名直接塞进全局配置：

```text
account_model_capability
  account_id, platform, workspace_id
  task_kind: chat | image
  upstream_model_id: 原始 slug / model_id
  display_name, aliases, metadata_json
  discovered_at, expires_at, last_verified_at
  state: available | unavailable | unknown | stale
  failure_reason
```

建议流程：

1. 账号新增、Cookie/token 更新、workspace 切换后，立即进行一次只读模型发现；失败时不覆盖最后一次成功快照。
2. 定时低频刷新，并在上游返回“模型不存在/无权使用”时只失效该账号 + 模型 + task_kind 记录，而不是把整个账号直接标为永久错误。
3. 管理端配置模型时提供三种可见范围：`所有健康账号可用（交集）`、`部分账号可用（并集，并显示数量）`、`自定义实验值（未验证）`。默认只允许选择交集。
4. 调度顺序改为：`active` + 未冷却 + 并发尚可 + 能力表中该模型/任务为 `available`，然后再按现有负载策略选账号。
5. 找不到候选账号时返回本地清晰错误（例如“模型仅在 1/4 个账号可用，当前该账号正忙”），不要将请求随机送至不支持的账号。
6. 成功调用更新 `last_verified_at` 并清除该模型能力的暂态错误；鉴权失败才影响整个账号状态。模型无权、配额耗尽、任务能力不符应是模型级/任务级状态。

## 对当前系统的最小落地顺序

1. 先为 GPT 的 `/backend-api/models` 与 Gemini 的 `client.list_models()` 增加“按账号读取并展示”的管理接口和内存/数据库快照，不改调用路径。
2. 将模型选项页由“静态 + 单账号/已初始化客户端的混合列表”改成明确的交集、并集、实验值三栏。
3. 再把账号池候选条件接入能力快照；只有此步骤完成后，全局模型配置才可以承诺“调度到的账号支持该模型”。
4. 豆包保持 `default`，直到实际网页协议存在可发现且可下传的模型选择字段。

## 参考来源

- [ChatGPT Web2API protocol reference：`/backend-api/models`](https://github.com/Octo-Lex/ChatGPT-Web2API/blob/master/docs/protocol-reference.md)
- [chat2api：网页请求模型映射](https://github.com/lanqian528/chat2api/blob/main/chatgpt/ChatService.py) 与 [README：多账号轮询 / 刷新策略](https://github.com/lanqian528/chat2api/blob/main/README.md)
- [yukkcat/chatgpt2api：统一模型目录源码](https://github.com/yukkcat/chatgpt2api/blob/main/services/model_catalog_service.py) 与 [3.2.3 目录策略变更](https://github.com/yukkcat/chatgpt2api/blob/main/CHANGELOG.md)
- [HanaokaYuzu/Gemini-API：动态模型发现与解析源码](https://github.com/HanaokaYuzu/Gemini-API/blob/main/src/gemini_webapi/client.py) 与 [README：动态发现说明](https://github.com/HanaokaYuzu/Gemini-API/blob/main/README.md#models)
- [Gemi2Api-Server：自定义模型 header 透传](https://github.com/zhiyu1998/Gemi2Api-Server)
