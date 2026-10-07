from __future__ import annotations

import json
import hashlib
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

    def _connect(self):
        return pymysql.connect(host=os.getenv("MYSQL_HOST", "127.0.0.1"),
                               port=int(os.getenv("MYSQL_PORT", "3306")),
                               user=os.getenv("MYSQL_USER", "root"),
                               password=os.getenv("MYSQL_PASSWORD", "123456"),
                               database=os.getenv("MYSQL_DATABASE", "flexi_admin"), charset="utf8mb4",
                               cursorclass=DictCursor, autocommit=True)

    def get_enabled_user(self, user_id: int) -> dict[str, Any] | None:
        """确认登录令牌对应的 Java 用户仍处于启用状态。"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id,username FROM sys_user WHERE id=%s AND enabled=1", (user_id,))
                return cursor.fetchone()

    @staticmethod
    def _proxy_ref(proxy_url: str) -> str:
        return hashlib.sha256(proxy_url.strip().encode()).hexdigest()

    def ensure_proxy_node(self, proxy_url: str, name: str = "") -> int | None:
        """旧账号携带代理 URL 时，将其归并为可管理的持久节点。"""
        proxy_url = proxy_url.strip()
        if not proxy_url:
            return None
        url_hash = self._proxy_ref(proxy_url)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO ai_proxy_node(name,proxy_url,url_hash) VALUES(%s,%s,%s) "
                    "ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)",
                    (name.strip() or f"legacy-{url_hash[:10]}", proxy_url, url_hash),
                )
                if cursor.lastrowid:
                    return int(cursor.lastrowid)
                cursor.execute("SELECT id FROM ai_proxy_node WHERE url_hash=%s", (url_hash,))
                return int(cursor.fetchone()["id"])

    def list_proxy_nodes(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""SELECT n.id,n.name,n.url_hash,n.status,n.failure_count,
                    COUNT(a.id) AS assigned_accounts FROM ai_proxy_node n
                    LEFT JOIN ai_proxy_account a ON a.proxy_id=n.id
                    GROUP BY n.id ORDER BY n.id""")
                return list(cursor.fetchall())

    def create_proxy_node(self, name: str, proxy_url: str) -> int:
        url_hash = self._proxy_ref(proxy_url)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO ai_proxy_node(name,proxy_url,url_hash) VALUES(%s,%s,%s)",
                               (name.strip(), proxy_url.strip(), url_hash))
                return int(cursor.lastrowid)

    def update_proxy_node(self, proxy_id: int, name: str, proxy_url: str, status: str) -> bool:
        url_hash = self._proxy_ref(proxy_url)
        with self._connect() as conn:
            # 节点和账号保留的旧 proxy 字段必须同时提交，避免重启后读到两个地址。
            conn.begin()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("UPDATE ai_proxy_node SET name=%s,proxy_url=%s,url_hash=%s,status=%s WHERE id=%s",
                                   (name.strip(), proxy_url.strip(), url_hash, status, proxy_id))
                    if cursor.rowcount == 0:
                        cursor.execute("SELECT id FROM ai_proxy_node WHERE id=%s", (proxy_id,))
                        if not cursor.fetchone():
                            conn.rollback()
                            return False
                    cursor.execute("UPDATE ai_proxy_account SET proxy=%s,credential_version=credential_version+1 WHERE proxy_id=%s",
                                   (proxy_url.strip(), proxy_id))
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def update_proxy_status(self, proxy_id: int, status: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE ai_proxy_node SET status=%s WHERE id=%s", (status, proxy_id))
                if cursor.rowcount:
                    return True
                cursor.execute("SELECT id FROM ai_proxy_node WHERE id=%s", (proxy_id,))
                return cursor.fetchone() is not None

    def delete_proxy_node(self, proxy_id: int) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM ai_proxy_account WHERE proxy_id=%s", (proxy_id,))
                if cursor.fetchone()["total"]:
                    raise ValueError("代理仍绑定账号，请先解绑后再删除")
                cursor.execute("DELETE FROM ai_proxy_node WHERE id=%s", (proxy_id,))
                return cursor.rowcount > 0

    def get_proxy_node(self, proxy_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id,name,proxy_url,status FROM ai_proxy_node WHERE id=%s", (proxy_id,))
                return cursor.fetchone()

    def record_proxy_failure(self, proxy_id: int) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE ai_proxy_node SET failure_count=failure_count+1,last_error='transport_failure' WHERE id=%s",
                               (proxy_id,))


    def import_account(self, platform: str, name: str, credentials: dict[str, Any], proxy: str = "", status: str = "active", cooldown_until: int = 0, failure_count: int = 0, error_message: str = "", proxy_id: int | None = None) -> dict[str, int]:
        # 明确的节点 ID 优先；旧版本账号配置继续通过原 URL 自动登记。
        if proxy_id:
            node = self.get_proxy_node(proxy_id)
            if not node:
                raise ValueError(f"Proxy node {proxy_id} not found")
            proxy = node["proxy_url"]
        elif proxy:
            proxy_id = self.ensure_proxy_node(proxy)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""INSERT INTO ai_proxy_account(platform,account_name,credentials,proxy,proxy_id,status,cooldown_until,failure_count,error_message)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE credentials=VALUES(credentials), credential_version=credential_version+1, proxy=VALUES(proxy), proxy_id=VALUES(proxy_id), status=VALUES(status), cooldown_until=VALUES(cooldown_until), failure_count=VALUES(failure_count), error_message=VALUES(error_message)""",
                    (platform, name, json.dumps(credentials, ensure_ascii=False), proxy, proxy_id, status, cooldown_until, failure_count, error_message))
                # 返回真实行身份与版本，使 Cookie 保存后的健康更新也使用正确 CAS。
                cursor.execute("SELECT id AS account_id,credential_version FROM ai_proxy_account WHERE platform=%s AND account_name=%s", (platform, name))
                return cursor.fetchone()

    def update_credentials(self, platform: str, name: str, patch: dict[str, Any], expected_version: int, account_id: int) -> bool:
        """原子保存轮换凭证；版本已变时不覆盖重新登录后的身份。"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ai_proxy_account SET credentials=JSON_MERGE_PATCH(credentials,%s), "
                    "credential_version=credential_version+1 WHERE platform=%s AND account_name=%s AND credential_version=%s AND id=%s",
                    (json.dumps(patch, ensure_ascii=False), platform, name, expected_version, account_id),
                )
                return cursor.rowcount == 1

    def save_account_health(self, platform: str, name: str, account: dict[str, Any]) -> None:
        """健康状态写入不回写凭证或代理；旧身份产生的状态不影响新登录。"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ai_proxy_account SET status=%s,cooldown_until=%s,failure_count=%s,error_message=%s "
                    "WHERE platform=%s AND account_name=%s AND credential_version=%s AND id=%s",
                    (account.get("status", "active"), account.get("cooldown_until", 0),
                     account.get("failure_count", 0), account.get("error_message", ""),
                     platform, name, account.get("credential_version", 0), account["account_id"]),
                )

    def update_supported_models(self, platform: str, name: str, models: list,
                                updated_at: int, expected_version: int, account_id: int) -> bool:
        """模型发现只更新元数据，不推进身份版本，避免使在途 OAuth 轮换 CAS 失效。"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ai_proxy_account SET credentials=JSON_MERGE_PATCH(credentials,%s) "
                    "WHERE platform=%s AND account_name=%s AND credential_version=%s AND id=%s",
                    (json.dumps({"supported_models": models, "models_updated_at": updated_at}, ensure_ascii=False),
                     platform, name, expected_version, account_id),
                )
                return cursor.rowcount == 1

    def update_account_cookie(self, platform: str, name: str, patch: dict,
                              expected_version: int, account_id: int) -> bool:
        """人工恢复同时保存新 Cookie 和健康状态，版本条件保护后台续期的新凭证。"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ai_proxy_account SET credentials=JSON_MERGE_PATCH(credentials,%s), "
                    "credential_version=credential_version+1,status='active',cooldown_until=0,"
                    "failure_count=0,error_message='' WHERE platform=%s AND account_name=%s "
                    "AND credential_version=%s AND id=%s",
                    (json.dumps(patch, ensure_ascii=False), platform, name, expected_version, account_id),
                )
                return cursor.rowcount == 1

    def bind_account_proxy(self, platform: str, name: str, proxy_id: int | None) -> dict[str, Any] | None:
        """代理变更也更新身份版本，阻止旧出口上的刷新覆盖新绑定。"""
        node = self.get_proxy_node(proxy_id) if proxy_id else None
        if proxy_id and (not node or node["status"] != "active"):
            raise ValueError("Proxy node not found or disabled")
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ai_proxy_account SET proxy=%s,proxy_id=%s,credential_version=credential_version+1 "
                    "WHERE platform=%s AND account_name=%s",
                    (node["proxy_url"] if node else "", proxy_id, platform, name),
                )
        rows = self.list_accounts(platform, name)
        return rows[0] if rows else None

    def list_accounts(self, platform: str, name: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                sql = """SELECT a.id AS account_id,a.account_name,a.credentials,a.credential_version,a.proxy,a.proxy_id,a.status,a.cooldown_until,a.failure_count,a.error_message,
                    n.proxy_url AS bound_proxy, n.status AS proxy_status FROM ai_proxy_account a LEFT JOIN ai_proxy_node n ON n.id=a.proxy_id
                    WHERE a.platform=%s"""
                if name is not None:
                    sql += " AND a.account_name=%s"
                cursor.execute(sql, (platform, name) if name is not None else (platform,))
                rows = cursor.fetchall()
        result = []
        for row in rows:
            if row["proxy"] and not row["proxy_id"]:
                proxy_id = self.ensure_proxy_node(row["proxy"])
                with self._connect() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("UPDATE ai_proxy_account SET proxy_id=%s WHERE platform=%s AND account_name=%s AND proxy_id IS NULL",
                                       (proxy_id, platform, row["account_name"]))
                node = self.get_proxy_node(proxy_id)
                row["proxy_id"] = proxy_id
                row["proxy_status"] = node["status"]
                row["bound_proxy"] = node["proxy_url"]
            credentials = json.loads(row["credentials"]) if isinstance(row["credentials"], str) else row["credentials"]
            # 已绑定节点以节点表为准；旧 proxy 列仅供尚未迁移的账号使用。
            proxy = row["bound_proxy"] if row["proxy_id"] else row["proxy"]
            credentials.update({"account_id": row["account_id"], "name" if platform != "gpt" else "email": row["account_name"], "credential_version": row.get("credential_version", 0), "proxy": proxy, "proxy_id": row["proxy_id"], "proxy_status": row["proxy_status"], "status": row["status"], "cooldown_until": row["cooldown_until"], "failure_count": row["failure_count"], "error_message": row["error_message"]})
            result.append(credentials)
        return result

    def delete_account(self, platform: str, name: str, account_id: int, expected_version: int) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM ai_proxy_account WHERE platform=%s AND account_name=%s AND id=%s AND credential_version=%s",
                               (platform, name, account_id, expected_version))
                return cursor.rowcount == 1


database = Database()
