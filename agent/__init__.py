import logging

from agent.base import BaseAgent, _VALID_MEMORY_TYPES
from llm_clients import VALID_LLM_PROVIDERS

logger = logging.getLogger(__name__)

VALID_MEMORY_TYPES = _VALID_MEMORY_TYPES


def _get_memory_type(config):
    params = getattr(config.agent, "params", None)
    if params is None:
        return "baseline"
    raw = getattr(params, "memory_type", "baseline")
    memory_type = str(raw).lower()
    if memory_type not in VALID_MEMORY_TYPES:
        logger.warning(
            "Invalid agent.params.memory_type %r; falling back to baseline. Valid: %s",
            raw,
            ", ".join(sorted(VALID_MEMORY_TYPES)),
        )
        return "baseline"
    return memory_type


def build_agent(config, system_prompt):
    memory_type = _get_memory_type(config)
    raw_provider = getattr(config.llm, "provider", "openai")
    provider = str(raw_provider).lower()
    if provider not in VALID_LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider {raw_provider!r}. "
            f"Choose from: {', '.join(sorted(VALID_LLM_PROVIDERS))}"
        )

    return BaseAgent(
        model=config.llm.name,
        seed=int(config.eval.seed),
        temperature=float(config.llm.temperature),
        timeout=float(config.llm.timeout),
        system_prompt=system_prompt,
        memory_type=memory_type,
        provider=provider,
    )
