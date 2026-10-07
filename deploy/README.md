# Python systemd 发布

服务器使用 Python 3.12、`uv` 和 `chatgpt2api.service`。代码发布到
`/opt/app/chatgpt2api/releases/`，`current` 指向运行版本；账号数据仍由 MySQL
保存；Gemini Cookie 续期结果写回 MySQL，抠图模型缓存在 `/var/lib/flexi-ai/`。
升级前确认账号表中的 Gemini Cookie 可用；旧版本留下的 `data/gemini_webapi` 缓存不再读取，账号验证通过后可清理。

首次发布前：

1. 将 `chatgpt2api.service` 安装到 `/etc/systemd/system/`，执行 `systemctl daemon-reload`。
2. 从 `chatgpt2api.env.example` 创建 `/etc/flexi/chatgpt2api.env`，填入 MySQL、Redis 和 OSS 配置。文件权限设为 `root:ubuntu 640`；密钥只保存在服务器，不提交到仓库。
3. 确认环境文件中的 MySQL 和 Redis 地址从宿主机可访问，并已由 Java Flyway 建立 `ai_proxy_*` 表。`REMBG_HOME` 目录须由运行服务的 `ubuntu` 用户写入。
4. 在 Python GitHub 仓库的 `production` 环境中配置 `SSH_PRIVATE_KEY`，使用服务器已授权的部署私钥。服务器地址、用户和已核验的主机公钥已写入工作流；服务器 SSH 主机密钥变更时需同步更新 `known_hosts`。

推送 `v*` 标签或手动运行 `deploy.yml` 后，工作流先运行测试，再上传源码并执行
`deploy.sh`。脚本安装锁定依赖、切换版本、重启服务，随后检查 `/health` 和
`/api/proxies`；检查失败会切回上一版本。

部署状态可用 `systemctl status chatgpt2api` 和 `journalctl -u chatgpt2api -f`
查看。服务只绑定 `127.0.0.1:8010`，由 Java 通过 `AI_PROXY_URL` 访问。


## 模型调用整改的发布顺序

1. 先由 Java Flyway 执行账号凭证版本和 dispatch 容量等待字段的迁移。当前版本分别为 `V4__gpt_credential_version.sql`、`V5__dispatch_capacity_wait.sql`；不能先启动依赖新字段的 Python 版本。
2. 在 `/etc/flexi/chatgpt2api.env` 明确配置 `GPT_LIMIT`、`GEMINI_LIMIT`、`DOUBAO_LIMIT`、`MODEL_GLOBAL_LIMIT`。这四项缺失或非正整数会拒绝启动。100/100/100/150 是模拟测试初值，生产值按服务器压测结果确定。
3. 设置 `GPT_REFRESH_CONCURRENCY=5`。它只限制 OAuth 刷新并发，与模型请求许可分开。
4. 更新 systemd unit 并执行 `systemctl daemon-reload`，使 Uvicorn 的 `--timeout-keep-alive 60` 生效。仅更新源码、重启旧 unit 不会改变该参数。
5. Java 启动命令在 `-jar` **之前**加入 `-Djdk.httpclient.keepalive.timeout=20`，并保持模型读取超时 300 秒。保留已有堆大小、GC 和其他 JVM 参数；不要用一整段新的 `JAVA_TOOL_OPTIONS` 覆盖已有配置。该属性作用于 JVM 内所有 JDK HttpClient，必须在客户端首次创建前设置。

Java 命令参数顺序示例（路径和现有内存参数使用服务器实际值）：

```sh
java <现有JVM参数> -Djdk.httpclient.keepalive.timeout=20 -jar /opt/app/backend/app.jar
```

Java 若由 `flexi-backend.service` 启动，应修改它实际使用的启动参数来源；发布工作流当前只重启服务，不会自动重写 unit 的 `ExecStart`。使用 `systemctl cat flexi-backend` 确认来源，修改后重新加载 unit。不要直接把示例中的占位符作为 shell 参数运行。

发布后检查 `/v1/capacity` 的平台与全局许可指标，并通过 Java 任务查询验证容量等待状态。20/60 秒是需要空闲边界压测验证的初值，不是消除所有 EOF 的保证。HTTP EOF 不自动重放模型 POST。
