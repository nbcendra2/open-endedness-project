from llm.base_client import BaseLLMClient

_PROVIDERS = {
    "openai": ("llm.openai_client", "OpenAIClient"),
    "deepseek": ("llm.deepseek_client", "DeepSeekClient"),
    "gemini": ("llm.gemini_client", "GeminiClient"),
}


def build_llm_client(provider: str = "openai", model: str | None = None) -> BaseLLMClient:
    """Factory — returns the right LLM client for the given provider.

    Args:
        provider: One of "openai", "deepseek", "gemini".
        model:    Model name (provider-specific). If ``None``, each client
                  uses its own default.
    """
    key = provider.lower()
    if key not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose from: {', '.join(sorted(_PROVIDERS))}"
        )

    module_path, class_name = _PROVIDERS[key]

    # lazy import so we don't pull in every SDK at startup
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    return cls(**kwargs)
