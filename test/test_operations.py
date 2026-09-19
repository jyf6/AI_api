from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import operations
from core.operations import begin_operation, replay_image


class FakeDatabase:
    def __init__(self):
        self.operations = {}

    def get_operation(self, operation_id):
        return self.operations.get(operation_id)

    def create_operation(self, operation_id, action):
        if operation_id in self.operations:
            return False
        self.operations[operation_id] = {
            "operation_id": operation_id, "action": action, "status": "RUNNING",
            "text_result": None, "image_result": None, "content_type": "", "error_message": "",
        }
        return True


def test_duplicate_operation_id_reuses_completed_image(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr("core.operations.database", database)

    operation_id, existing = begin_operation("operation-1", "image")
    assert operation_id == "operation-1"
    assert existing is None

    database.operations[operation_id].update(status="SUCCEEDED", image_result=b"png", content_type="image/png")
    _, existing = begin_operation(operation_id, "image")

    response = replay_image(existing, operation_id)
    assert response.body == b"png"
    assert response.headers["x-operation-id"] == operation_id


def test_operation_query_returns_saved_image(monkeypatch):
    database = FakeDatabase()
    database.operations["operation-2"] = {
        "operation_id": "operation-2", "action": "image", "status": "SUCCEEDED",
        "text_result": None, "image_result": b"png", "content_type": "image/png", "error_message": "",
    }
    monkeypatch.setattr(operations, "database", database)
    app = FastAPI()
    app.include_router(operations.router)

    response = TestClient(app).get("/v1/operations/operation-2")

    assert response.status_code == 200
    assert response.content == b"png"
