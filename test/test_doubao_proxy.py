from __future__ import annotations

import unittest

from providers.doubao.backend import DoubaoBackendAPI


class DoubaoProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_socks5h_uses_remote_dns(self) -> None:
        backend = DoubaoBackendAPI({"sessionid": "test"}, "socks5h://user:pass@proxy.example:1080")

        connector = backend._build_connector()
        try:
            self.assertEqual(connector._proxy_type.name, "SOCKS5")
            self.assertEqual(connector._proxy_host, "proxy.example")
            self.assertEqual(connector._proxy_port, 1080)
            self.assertEqual(connector._proxy_username, "user")
            self.assertEqual(connector._proxy_password, "pass")
            self.assertTrue(connector._rdns)
        finally:
            await connector.close()

    async def test_socks5_keeps_existing_dns_behavior(self) -> None:
        backend = DoubaoBackendAPI({"sessionid": "test"}, "socks5://proxy.example:1080")

        connector = backend._build_connector()
        try:
            self.assertFalse(connector._rdns)
        finally:
            await connector.close()


if __name__ == "__main__":
    unittest.main()
