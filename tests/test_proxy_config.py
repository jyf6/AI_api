from core.database import Database


def test_proxy_reference_is_stable_and_distinguishes_credentials():
    first = "socks5h://user:pass@proxy.example:1080"
    assert Database._proxy_ref(first) == Database._proxy_ref(first)
    assert Database._proxy_ref(first) != Database._proxy_ref("socks5h://other:pass@proxy.example:1080")
