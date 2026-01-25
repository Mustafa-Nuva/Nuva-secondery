from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAIEngine(ABC):
    """Abstract interface for AI engines.

    Implementations should return a short, polite reply in Kurdish (Sorani).
    """

    @abstractmethod
    def generate_reply(self, user_text: str) -> str:
        raise NotImplementedError
