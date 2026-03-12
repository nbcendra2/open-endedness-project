from agent.random_agent import RandomAgent
from agent.base import BaseAgent
from agent.memory_agent import MemoryAgent


def build_agent(config, system_prompt):
    agent_name = str(config.agent.name).lower()

    if agent_name == "random":
        return RandomAgent(seed=int(config.eval.seed))

    if agent_name == "base":
        return BaseAgent(
            model=config.openai_model.name,
            seed=int(config.eval.seed),
            temperature=float(config.openai_model.temperature),
            timeout=float(config.openai_model.timeout),
            system_prompt=system_prompt,
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
        )

    raise ValueError(f"Unsupported agent.name: {config.agent.name}")