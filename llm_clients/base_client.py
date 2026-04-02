"""Functionality: Define the abstract interface every LLM backend must implement

Concrete clients share the same four methods and return shapes so callers can
swap providers without changing call sites
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLMClient(ABC):
    """Abstract base for all LLM provider clients

    Every provider must implement these four methods and return
    the same Python types so that callers remain provider-agnostic
    """

    @abstractmethod
    def generate(self, messages: List[Dict], temperature: float = 0.2, timeout: float = 10) -> str:
        """Free-form text generation; returns a plain string"""
        ...

    @abstractmethod
    def generate_action_structured(
        self, messages: List[Dict], valid_actions: List[str],
        temperature: float = 0.2, timeout: float = 10,
    ) -> dict:
        """Return dict with keys reason and action where action is in valid_actions"""
        ...

    @abstractmethod
    def generate_planning_structured(
        self, messages: List[Dict], temperature: float = 0.3, timeout: float = 15,
    ) -> dict:
        """Return dict with key plan (string)"""
        ...

    @abstractmethod
    def generate_reflection(
        self, messages: List[Dict], temperature: float = 0.3, timeout: float = 30,
    ) -> dict:
        """Return dict with keys summary, strategy, lessons (list of strings)"""
        ...

    def embed(self, text: str) -> List[float]:
        """Return embedding vector for text. Default: OpenAI text-embedding-3-small"""
        import os, openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
