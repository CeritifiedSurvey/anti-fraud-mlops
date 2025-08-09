from typing import Callable, Dict, Tuple, Optional


class FastAPI:
    """A very small subset of FastAPI's interface.

    It supports registering POST routes and a single startup event so that the
    rest of the project can be exercised without the real FastAPI dependency.
    """

    def __init__(self) -> None:
        self.routes: Dict[Tuple[str, str], Callable] = {}
        self.startup: Optional[Callable[[], None]] = None

    def on_event(self, name: str) -> Callable[[Callable], Callable]:
        def decorator(func: Callable) -> Callable:
            if name == "startup":
                self.startup = func
            return func
        return decorator

    def post(self, path: str) -> Callable[[Callable], Callable]:
        def decorator(func: Callable) -> Callable:
            self.routes[(path, "POST")] = func
            return func
        return decorator


__all__ = ["FastAPI"]
