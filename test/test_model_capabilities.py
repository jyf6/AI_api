from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import accounts


class FakeAccountPool:
    def __init__(self):
        self.accounts = [
            {"email": "one@example.com", "status": "active", "supported_models": [{"value": "shared"}, {"value": "one"}]},
            {"email": "two@example.com", "status": "active", "supported_models": [{"value": "shared"}, {"value": "two"}]},
        ]

    def list_accounts(self):
        return self.accounts

    def refresh_supported_models(self, email):
        return next(account["supported_models"] for account in self.accounts if account["email"] == email)


def test_model_capabilities_exposes_each_account_and_intersection(monkeypatch):
    pool = FakeAccountPool()
    monkeypatch.setattr(accounts, "account_service", pool)
    app = FastAPI()
    app.include_router(accounts.router)
    client = TestClient(app)

    refreshed = client.post("/api/model-capabilities/gpt/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["results"]["one@example.com"]["status"] == "ok"

    response = client.get("/api/model-capabilities/gpt")
    assert response.status_code == 200
    assert response.json()["intersection"] == ["shared"]
    assert len(response.json()["accounts"]) == 2


def test_gemini_model_options_only_use_live_registry(monkeypatch):
    live_models = [{"value": "gemini-flash", "label": "Gemini Flash", "description": ""}]
    monkeypatch.setattr(accounts.gemini_account_service, "get_available_models", lambda: live_models)
    monkeypatch.setattr(accounts.account_service, "get_available_models", lambda: [])

    options = __import__("asyncio").run(accounts.get_model_options())

    assert options["gemini"]["image_models"] == live_models
    assert options["gemini"]["chat_models"] == live_models
    assert "gemini-ultra" not in {item["value"] for item in options["gemini"]["chat_models"]}
