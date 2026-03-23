from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLMClient(ABC):
    """Abstract base for all LLM provider clients.

    Every provider must implement these four methods and return
    the same Python types so that callers remain provider-agnostic.
    """

    @abstractmethod
    def generate(self, messages: List[Dict], temperature: float = 0.2, timeout: float = 10) -> str:
        """Free-form text generation.  Returns a plain string."""
        ...

    @abstractmethod
    def generate_action_structured(
        self, messages: List[Dict], valid_actions: List[str],
        temperature: float = 0.2, timeout: float = 10,
    ) -> dict:
        """Return ``{"reason": str, "action": str}`` where *action* is in *valid_actions*."""
        ...

    @abstractmethod
    def generate_planning_structured(
        self, messages: List[Dict], temperature: float = 0.3, timeout: float = 15,
    ) -> dict:
        """Return ``{"plan": str}``."""
        ...

    @abstractmethod
    def generate_reflection(
        self, messages: List[Dict], temperature: float = 0.3, timeout: float = 30,
    ) -> dict:
        """Return ``{"summary": str, "strategy": str, "lessons": List[str]}``."""
        ...
