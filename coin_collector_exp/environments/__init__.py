"""Public entry point for creating wrapped environments (BabyAI or TextWorld).

Usage:
    from environments import make_env

    # BabyAI
    env = make_env(env_name="BabyAI-GoToLocal-v0")

    # TextWorld Coin Collector
    env = make_env(env_name="tw-coin_collector-L5")
"""

from environments.env_wrapper import EnvState, EnvWrapper, StepResult


def make_env(
    env_name="BabyAI-MixedTrainLocal-v0",
    gym_kwargs=None,
    invalid_action_mode="fallback",
    fallback_action="go forward",
):
    """Create an environment wrapped for LLM agents.

    Routing logic:
        - Names starting with ``tw-`` → TextWorldEnvWrapper
        - Everything else             → EnvWrapper (BabyAI / Minigrid)

    Args:
        env_name:
            Environment identifier.
            BabyAI examples : ``BabyAI-GoToLocal-v0``, ``BabyAI-MixedTrainLocal-v0/goto``
            TextWorld examples: ``tw-coin_collector-L1``, ``tw-coin_collector-L10``
        gym_kwargs:
            Optional kwargs forwarded to ``gym.make`` (BabyAI only).
        invalid_action_mode:
            ``'strict'`` raises on bad actions; ``'fallback'`` silently substitutes.
        fallback_action:
            Replacement action under fallback mode.
            BabyAI default: ``'go forward'``.  TextWorld default: ``'look'``.

    Returns:
        EnvWrapper or TextWorldEnvWrapper — both expose identical
        ``reset → EnvState`` / ``step → StepResult`` interfaces.
    """
    if env_name.lower().startswith("tw-"):
        from environments.tw_env_wrapper import TextWorldEnvWrapper

        # "go forward" is a BabyAI action; swap to a safe TextWorld no-op
        tw_fallback = fallback_action if fallback_action != "go forward" else "look"
        return TextWorldEnvWrapper(
            env_name=env_name,
            invalid_action_mode=invalid_action_mode,
            fallback_action=tw_fallback,
            gym_kwargs=gym_kwargs,
        )

    return EnvWrapper(
        env_name=env_name,
        gym_kwargs=gym_kwargs,
        invalid_action_mode=invalid_action_mode,
        fallback_action=fallback_action,
    )