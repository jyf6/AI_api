import ast
from pathlib import Path
import unittest


class ApiRoutesWithoutAuthTest(unittest.TestCase):
    def test_business_routes_have_no_auth_dependencies(self):
        project_root = Path(__file__).resolve().parents[1]
        app_tree = ast.parse((project_root / "api" / "app.py").read_text(encoding="utf-8"))
        health_tree = ast.parse((project_root / "api" / "routers" / "health.py").read_text(encoding="utf-8"))
        router_registrations = [
            node for node in ast.walk(app_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "include_router"
        ]

        self.assertTrue(router_registrations)
        self.assertTrue(all(not any(keyword.arg == "dependencies" for keyword in node.keywords) for node in router_registrations))
        self.assertFalse(any(isinstance(node, ast.Name) and node.id == "Depends" for node in ast.walk(health_tree)))


if __name__ == "__main__":
    unittest.main()
