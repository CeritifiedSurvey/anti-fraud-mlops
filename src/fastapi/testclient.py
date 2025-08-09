from typing import Any, Dict


class Response:
    def __init__(self, data: Dict[str, Any], status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._data


class TestClient:
    """A minimal test client for the local FastAPI stub."""

    def __init__(self, app):
        self.app = app
        if getattr(app, "startup", None):
            app.startup()

    def post(self, path: str, json: Dict[str, Any]) -> Response:
        handler = self.app.routes[(path, "POST")]
        data = handler(json)
        return Response(data)


__all__ = ["TestClient"]
