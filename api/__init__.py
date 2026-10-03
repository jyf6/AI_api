def create_app():
    """延迟加载服务路由，避免导入独立 API 工具时初始化全部模型依赖。"""
    from api.app import create_app as app_factory

    return app_factory()

__all__ = ["create_app"]

