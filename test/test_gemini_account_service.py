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


def test_cookie_header_with_equals_and_special_characters_is_parsed():
    from services.gemini_account_service import parse_cookie_header

    cookies = parse_cookie_header("SID=g.a/x=y; __Secure-1PSID=g.a/x=y; SAPISID=a/b; __Secure-1PSIDTS=sidts-x_y")
    assert cookies["__Secure-1PSID"] == "g.a/x=y"
    assert cookies["SAPISID"] == "a/b"


def test_gemini_account_requires_psidts(monkeypatch, tmp_path):
    monkeypatch.setattr("services.gemini_account_service.ACCOUNTS_FILE", tmp_path / "gemini_accounts.json")
    service = GeminiAccountService()
    try:
        service.add_account("missing-ts", "__Secure-1PSID=psid")
        assert False
    except ValueError as exc:
        assert "PSIDTS" in str(exc)


def test_gemini_image_alias_maps_to_registered_runtime_model():
    from services.gemini_backend_api import _runtime_model

    assert _runtime_model("gemini-2.5-pro-image") == "gemini-pro"
    assert _runtime_model("gemini-3-flash") == "gemini-3-flash"
