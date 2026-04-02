"""Functionality: Expose build_llm_client so the rest of the codebase gets a
BaseLLMClient without importing OpenAI, Gemini or DeepSeek directly

Call build_llm_client to obtain the right implementation; the agent only
depends on that interface
"""

from llm_clients.base_client import BaseLLMClient

# Map provider name to (module path, class name) for lazy loading
_PROVIDERS = {
    "openai": ("llm_clients.openai_client", "OpenAIClient"),
    "deepseek": ("llm_clients.deepseek_client", "DeepSeekClient"),
    "gemini": ("llm_clients.gemini_client", "GeminiClient"),
}

VALID_LLM_PROVIDERS = frozenset(_PROVIDERS)


def build_llm_client(provider: str = "openai", model: str | None = None) -> BaseLLMClient:
    """Factory: returns the right LLM client for the given provider

    Args:
        provider: One of "openai", "deepseek", "gemini"
        model:    Model name (provider-specific); if None, each client uses its default
    """
    key = provider.lower()
    if key not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose from: {', '.join(sorted(_PROVIDERS))}"
        )

    module_path, class_name = _PROVIDERS[key]

    # Lazy import: load only the SDK for the chosen provider (faster startup, fewer missing-deps issues)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    return cls(**kwargs)
