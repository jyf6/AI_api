from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Database:
    """Python 代理账户和模型配置的唯一 MySQL 数据访问入口。"""

    def __init__(self, path: Path = DATA_DIR / "accounts.db") -> None:
        self.path = path  # 保留参数以兼容旧脚本，SQLite 不再使用。
        if os.getenv("STORAGE_BACKEND", "mysql").lower() != "mysql":
            raise RuntimeError("账户存储只支持 MySQL，请设置 STORAGE_BACKEND=mysql")

    def _connect(self):
        return pymysql.connect(host=os.getenv("MYSQL_HOST", "127.0.0.1"), port=int(os.getenv("MYSQL_PORT", "3306")),
                               user=os.getenv("MYSQL_USER", "root"), password=os.getenv("MYSQL_PASSWORD", "123456"),
                               database=os.getenv("MYSQL_DATABASE", "flexi_admin"), charset="utf8mb4",
                               cursorclass=DictCursor, autocommit=True)

    def get_model(self, platform: str, action: str) -> str:
        column = "image_model" if action == "image" else "chat_model"
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT {column} AS model FROM ai_proxy_model_config WHERE platform=%s", (platform,))
                row = cursor.fetchone()
        return str(row["model"]) if row else ""

    def list_model_configs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT platform,image_model,chat_model,description,updated_at FROM ai_proxy_model_config ORDER BY platform")
                return list(cursor.fetchall())

    def update_model_config(self, platform: str, image_model: str, chat_model: str, description: str = "") -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""INSERT INTO ai_proxy_model_config(platform,image_model,chat_model,description) VALUES(%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE image_model=VALUES(image_model), chat_model=VALUES(chat_model), description=IF(VALUES(description)='',description,VALUES(description))""",
                    (platform, image_model.strip(), chat_model.strip(), description.strip()))

    def import_account(self, platform: str, name: str, credentials: dict[str, Any], proxy: str = "", status: str = "active", cooldown_until: int = 0, failure_count: int = 0, error_message: str = "") -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""INSERT INTO ai_proxy_account(platform,account_name,credentials,proxy,status,cooldown_until,failure_count,error_message)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE credentials=VALUES(credentials), proxy=VALUES(proxy), status=VALUES(status), cooldown_until=VALUES(cooldown_until), failure_count=VALUES(failure_count), error_message=VALUES(error_message)""",
                    (platform, name, json.dumps(credentials, ensure_ascii=False), proxy, status, cooldown_until, failure_count, error_message))

    def list_accounts(self, platform: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT account_name,credentials,proxy,status,cooldown_until,failure_count,error_message FROM ai_proxy_account WHERE platform=%s", (platform,))
                rows = cursor.fetchall()
        result = []
        for row in rows:
            credentials = json.loads(row["credentials"]) if isinstance(row["credentials"], str) else row["credentials"]
            credentials.update({"name" if platform != "gpt" else "email": row["account_name"], "proxy": row["proxy"], "status": row["status"], "cooldown_until": row["cooldown_until"], "failure_count": row["failure_count"], "error_message": row["error_message"]})
            result.append(credentials)
        return result

    def delete_account(self, platform: str, name: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM ai_proxy_account WHERE platform=%s AND account_name=%s", (platform, name))

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT operation_id,action,status,text_result,image_result,content_type,error_message FROM ai_proxy_operation WHERE operation_id=%s", (operation_id,))
                return cursor.fetchone()

    def create_operation(self, operation_id: str, action: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                return cursor.execute("INSERT IGNORE INTO ai_proxy_operation(operation_id,action,status) VALUES(%s,%s,'RUNNING')", (operation_id, action)) == 1

    def complete_operation(self, operation_id: str, text: str | None = None, image: bytes | None = None, content_type: str = "") -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE ai_proxy_operation SET status='SUCCEEDED',text_result=%s,image_result=%s,content_type=%s,error_message='' WHERE operation_id=%s", (text, image, content_type, operation_id))

    def fail_operation(self, operation_id: str, error: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE ai_proxy_operation SET status='SENT_UNKNOWN',error_message=%s WHERE operation_id=%s", (error[:500], operation_id))

    def recover_running_operations(self) -> None:
        """进程重启后，未完成调用只能判为已发送未知，不能自动再次提交。"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE ai_proxy_operation SET status='SENT_UNKNOWN',error_message='代理重启，需查询上游结果' WHERE status='RUNNING'")


database = Database()
