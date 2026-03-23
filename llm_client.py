"""Backward-compatible entry point.

Existing code that does ``from llm_client import LLMClient`` will keep
working — ``LLMClient`` is now an alias for :class:`OpenAIClient`.

For new code, prefer::

    from llm import build_llm_client
    client = build_llm_client(provider="openai", model="gpt-4o-mini")
"""

from llm import build_llm_client  # noqa: F401
from llm.openai_client import OpenAIClient as LLMClient  # noqa: F401
