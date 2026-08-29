from services.gemini_account_service import GeminiAccountService


def test_gemini_account_requires_psid_and_degrades_auth_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("services.gemini_account_service.ACCOUNTS_FILE", tmp_path / "gemini_accounts.json")
    service = GeminiAccountService()

    try:
        service.add_account("bad", "SID=x")
        assert False, "missing __Secure-1PSID must be rejected"
    except ValueError:
        pass

    service.add_account("main", "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")
    account = service.get_available_account()
    service.release_account(account["name"], False, "Gemini auth cookie expired (401)")

    assert service.list_accounts()[0]["status"] == "error"
