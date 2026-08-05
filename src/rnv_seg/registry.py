from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in {self.name}")
            self._items[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._items:
            choices = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.name} {name!r}. Available: {choices}")
        return self._items[name](**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


MODELS = Registry("model")
DATASETS = Registry("dataset")
LOSSES = Registry("loss")

