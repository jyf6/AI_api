-- =============================================================
-- ChatGPT-Image-Service 数据库建表语句
-- 服务启动不再自动建表，请在部署前手动执行本文件初始化表结构。
-- 所有建表语句均为 IF NOT EXISTS，可安全重复执行。
-- 连接参数（库名等）以实际部署环境为准。
-- =============================================================

-- 模型配置表：每个平台一行，分别存生图模型名与聊天模型名
CREATE TABLE IF NOT EXISTS ai_proxy_model_config (
    platform VARCHAR(16) PRIMARY KEY,
    image_model VARCHAR(128) NOT NULL DEFAULT '',
    chat_model VARCHAR(128) NOT NULL DEFAULT '',
    description VARCHAR(500) NOT NULL DEFAULT '',
    updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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

-- 操作记录表：用于请求幂等与崩溃恢复
CREATE TABLE IF NOT EXISTS ai_proxy_operation (
    operation_id CHAR(36) NOT NULL PRIMARY KEY,
    action VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL,
    text_result LONGTEXT NULL,
    image_result LONGBLOB NULL,
    content_type VARCHAR(100) NOT NULL DEFAULT '',
    error_message VARCHAR(500) NOT NULL DEFAULT '',
    created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    KEY idx_ai_proxy_operation_status_updated(status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
