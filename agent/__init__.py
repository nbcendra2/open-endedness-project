from agent.random_agent import RandomAgent
from agent.base import BaseAgent
from agent.memory_agent import MemoryAgent


def build_agent(config, system_prompt):
    agent_type = str(config.agent.type).lower()

    if agent_type == "random":
        return RandomAgent(seed=int(config.eval.seed))

    if agent_type == "base":
        return BaseAgent(
            model=config.openai_model.name,
            seed=int(config.eval.seed),
            temperature=float(config.openai_model.temperature),
            timeout=float(config.openai_model.timeout),
            system_prompt=system_prompt,
        )

    if agent_type == "memory":
        return MemoryAgent(
            model=config.openai_model.name,
            seed=int(config.eval.seed),
            temperature=float(config.openai_model.temperature),
            timeout=float(config.openai_model.timeout),
            system_prompt=system_prompt,
            memory_path=config.memory.path,
            retrieval_top_k=int(config.memory.retrieval_top_k),
        )

    raise ValueError(f"Unsupported agent.type: {config.agent.type}")