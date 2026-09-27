-- =============================================================
-- ChatGPT-Image-Service 数据库建表语句
-- 服务启动不再自动建表，请在部署前手动执行本文件初始化表结构。
-- 所有建表语句均为 IF NOT EXISTS，可安全重复执行。
-- 连接参数（库名等）以实际部署环境为准。
-- =============================================================


-- 账号表：credentials 为 JSON，兼容各平台不同凭证结构
CREATE TABLE IF NOT EXISTS ai_proxy_account (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    platform VARCHAR(16) NOT NULL,
    account_name VARCHAR(255) NOT NULL,
    credentials JSON NOT NULL,
    proxy VARCHAR(500) NOT NULL DEFAULT '',
    status VARCHAR(24) NOT NULL DEFAULT 'active',
    cooldown_until BIGINT NOT NULL DEFAULT 0,
    failure_count INT NOT NULL DEFAULT 0,
    error_message VARCHAR(500) NOT NULL DEFAULT '',
    created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    UNIQUE KEY uk_ai_proxy_account_platform_name(platform, account_name),
    KEY idx_ai_proxy_account_platform_status(platform, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
