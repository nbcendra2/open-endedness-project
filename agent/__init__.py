from agent.random_agent import RandomAgent
from agent.base import BaseAgent
from agent.memory_agent import MemoryAgent


VALID_MEMORY_TYPES = {
    "baseline",
    "trajectory",
    "reflection",
    "enriched",
    "fade_enriched",
    "semantic_enriched",
}


def _get_memory_type(config):
    params = getattr(config.agent, "params", None)
    if params is None:
        return "baseline"
    memory_type = str(getattr(params, "memory_type", "baseline")).lower()
    if memory_type not in VALID_MEMORY_TYPES:
        # Fallback to baseline for now; later we can raise if desired
        return "baseline"
    return memory_type


def build_agent(config, system_prompt):
    agent_name = str(config.agent.name).lower()
    memory_type = _get_memory_type(config)

    if agent_name == "random":
        return RandomAgent(seed=int(config.eval.seed))

    if agent_name == "base":
        return BaseAgent(
            model=config.openai_model.name,
            seed=int(config.eval.seed),
            temperature=float(config.openai_model.temperature),
            timeout=float(config.openai_model.timeout),
            system_prompt=system_prompt,
            memory_type=memory_type,
        )

    if agent_name == "memory":
        params = config.agent.params
        return MemoryAgent(
            model=config.openai_model.name,
            seed=int(config.eval.seed),
            temperature=float(config.openai_model.temperature),
            timeout=float(config.openai_model.timeout),
            system_prompt=system_prompt,
            memory_path=str(params.memory_path),
            retrieval_top_k=int(params.retrieval_top_k),
            retriever=str(params.retriever),
            stuck_window=int(params.stuck_window),
            reflection=bool(params.reflection),
            planning=bool(params.planning),
            memory_type=memory_type,
        )

    raise ValueError(f"Unsupported agent.name: {config.agent.name}")