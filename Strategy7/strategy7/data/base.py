"""Base interfaces for data components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import MarketBundle


class MarketDataLoader(ABC):
    """Load raw market data (daily + minute)."""

    @abstractmethod
    def load(self) -> MarketBundle:
        raise NotImplementedError

