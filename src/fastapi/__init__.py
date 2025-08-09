class FastAPI:
    """A very small subset of the FastAPI interface used for tests."""

    def __init__(self) -> None:
        self.routes = {}
        self.startup_handler = None

    def post(self, path: str):
        def decorator(func):
            self.routes[("POST", path)] = func
            return func

        return decorator

    def on_event(self, name: str):
        def decorator(func):
            if name == "startup":
                self.startup_handler = func
            return func

        return decorator
