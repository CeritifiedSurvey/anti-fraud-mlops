class _Response:
    def __init__(self, data):
        self.status_code = 200
        self._data = data

    def json(self):
        return self._data


class TestClient:
    """Minimal test client executing handlers directly."""

    def __init__(self, app):
        self.app = app
        if getattr(app, "startup_handler", None):
            app.startup_handler()

    def post(self, path: str, json=None):
        handler = self.app.routes.get(("POST", path))
        if handler is None:
            raise ValueError(f"No POST handler for {path}")
        result = handler(json)
        return _Response(result)
