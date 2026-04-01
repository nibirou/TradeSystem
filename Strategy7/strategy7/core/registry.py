"""Simple registry utility for plugin architecture."""

from __future__ import annotations

from typing import Dict, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic name -> object registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, obj: T, overwrite: bool = True) -> None:
        k = str(key).strip()
        if not k:
            raise ValueError(f"{self.name} registry key cannot be empty.")
        if (k in self._items) and (not overwrite):
            raise KeyError(f"{self.name} registry already has key: {k}")
        self._items[k] = obj

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(f"{self.name} registry missing key: {key}. available={self.keys()}")
        return self._items[key]

    def keys(self) -> List[str]:
        return sorted(self._items.keys())

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

