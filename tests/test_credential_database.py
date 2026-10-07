from core.database import Database


class Connection:
    def __init__(self, affected):
        self.rowcount = affected
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def cursor(self):
        return self

    def execute(self, sql, parameters):
        self.calls.append((sql, parameters))


def test_credential_write_has_version_condition_and_preserves_unrelated_json(monkeypatch):
    db = Database()
    connection = Connection(1)
    monkeypatch.setattr(db, "_connect", lambda: connection)
    assert db.update_credentials("gpt", "one", {"access_token": "new", "refresh_token": "rotated"}, 7, 11)
    sql, parameters = connection.calls[0]
    assert "JSON_MERGE_PATCH" in sql
    assert "credential_version=credential_version+1" in sql
    assert "AND credential_version=%s" in sql
    assert "AND id=%s" in sql
    assert parameters[-4:] == ("gpt", "one", 7, 11)
    connection.rowcount = 0
    assert not db.update_credentials("gpt", "one", {"access_token": "old-result"}, 7, 11)


def test_health_write_never_contains_credentials_or_proxy_columns(monkeypatch):
    db = Database()
    connection = Connection(1)
    monkeypatch.setattr(db, "_connect", lambda: connection)
    db.save_account_health("gpt", "one", {"account_id": 11, "status": "error", "credential_version": 7,
        "access_token": "stale", "refresh_token": "stale", "proxy": "old"})
    sql, parameters = connection.calls[0]
    assert "credentials=" not in sql
    assert "proxy=" not in sql
    assert parameters[-2:] == (7, 11)


def test_cookie_import_returns_row_identity_and_new_version(monkeypatch):
    db=Database()
    connection=Connection(1)
    connection.fetchone=lambda: {"account_id":11,"credential_version":8}
    monkeypatch.setattr(db,"_connect",lambda:connection)
    identity=db.import_account("gemini","one",{"psid":"new-cookie"})
    assert identity == {"account_id":11,"credential_version":8}
    assert "credential_version=credential_version+1" in connection.calls[0][0]
    assert "id AS account_id,credential_version" in connection.calls[1][0]


def test_model_metadata_does_not_advance_credential_identity(monkeypatch):
    db = Database()
    connection = Connection(1)
    monkeypatch.setattr(db, "_connect", lambda: connection)
    assert db.update_supported_models("gpt", "one", [{"value": "model"}], 123, 7, 11)
    sql, parameters = connection.calls[0]
    assert "credential_version=credential_version+1" not in sql
    assert "AND credential_version=%s AND id=%s" in sql
    assert parameters[-2:] == (7, 11)


def test_account_deletion_is_scoped_to_original_identity(monkeypatch):
    db = Database()
    connection = Connection(1)
    monkeypatch.setattr(db, "_connect", lambda: connection)
    assert db.delete_account("gpt", "one", 11, 7)
    sql, parameters = connection.calls[0]
    assert "AND id=%s AND credential_version=%s" in sql
    assert parameters == ("gpt", "one", 11, 7)
    connection.rowcount = 0
    assert not db.delete_account("gpt", "one", 11, 7)
