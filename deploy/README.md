# Python systemd 发布

服务器使用 Python 3.12、`uv` 和 `chatgpt2api.service`。代码发布到
`/opt/app/chatgpt2api/releases/`，`current` 指向运行版本；账号数据仍由 MySQL
保存，`data`、Gemini Cookie 缓存和抠图模型缓存位于 `/var/lib/flexi-ai/`。

首次发布前：

1. 将 `chatgpt2api.service` 安装到 `/etc/systemd/system/`，执行 `systemctl daemon-reload`。
2. 从 `chatgpt2api.env.example` 创建 `/etc/flexi/chatgpt2api.env`。MySQL 与 Redis 连接参数直接写在 `core/database.py` 和 `api/app.py` 中，须与服务器 Docker Compose 一致。
3. 确认本机 `127.0.0.1:3306` 和 `127.0.0.1:6379` 可用，并已由 Java Flyway 建立 `ai_proxy_*` 表。
4. 在 Python GitHub 仓库的 `production` 环境中配置 `SSH_PRIVATE_KEY`，使用服务器已授权的部署私钥。服务器地址、用户和已核验的主机公钥已写入工作流；服务器 SSH 主机密钥变更时需同步更新 `known_hosts`。

推送 `v*` 标签或手动运行 `deploy.yml` 后，工作流先运行测试，再上传源码并执行
`deploy.sh`。脚本安装锁定依赖、切换版本、重启服务，随后检查 `/health` 和
`/api/proxies`；检查失败会切回上一版本。

部署状态可用 `systemctl status chatgpt2api` 和 `journalctl -u chatgpt2api -f`
查看。服务只绑定 `127.0.0.1:8010`，由 Java 通过 `AI_PROXY_URL` 访问。
