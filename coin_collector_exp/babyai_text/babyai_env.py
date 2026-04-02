"""
Functionalities:

    It builds BabyAI Gym environments and wraps them for **text-based agents**: the game
    still runs in Gymnasium, but actions are English strings and observations carry text
    (and a POV image). The list ``BABYAI_ENVS`` names tasks you can use; ``make_babyai_env``
    creates one env and can target a specific MixedTrain mission kind (``…/goto`` etc.).

    Most experiments here go through ``environments.make_env`` / ``EnvWrapper``; use this
    module when you already have ``config.envs.babyai_kwargs`` on your config object.

Implementation detail: ``make_babyai_env`` applies ``BabyAITextCleanLangWrapper`` so
``gym.make`` kwargs and wrapper kwargs stay in sync via ``config.envs.babyai_kwargs``.
"""

from typing import Optional

import gymnasium as gym
import minigrid

from babyai_text import BabyAITextCleanLangWrapper

# Minigrid must be imported so that BabyAI env specs are registered with Gymnasium.
# Without registration, ``gym.make("BabyAI-...")`` may fail with an unknown env id.
minigrid.register_minigrid_envs()

# Bonus BabyAI levels that were reported broken or unstable in certain Minigrid releases.
# We skip them when building the global list of available BabyAI task names.
# Upstream context: https://github.com/Farama-Foundation/Minigrid/pull/381#issuecomment-1646800992
broken_bonus_envs = {
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-KeyInBox-v0",
}

# At import time, collect every registered Gym environment whose id starts with "BabyAI",
# excluding the broken bonus ids above. This list is handy for menus, tests, or sweeps.
BABYAI_ENVS = []
for env_spec in gym.envs.registry:
    id = env_spec
    if id.split("-")[0] == "BabyAI":
        if id not in broken_bonus_envs:
            BABYAI_ENVS.append(id)

# These entries are NOT registered Gymnasium ids. They are project-specific strings:
#   "BabyAI-MixedTrainLocal-v0/<suffix>"
# meaning: create the real env ``BabyAI-MixedTrainLocal-v0``, then resample (see
# ``make_babyai_env``) until the mission family matches ``<suffix>``. The suffix must
# match ``env.unwrapped.action_kinds[0]`` after replacing spaces with underscores
# (e.g. "go to" becomes "goto"). This lets you benchmark one mission type from a mixed
# distribution without mixing others in the same run configuration string.
BABYAI_ENVS += [
    "BabyAI-MixedTrainLocal-v0/goto",
    "BabyAI-MixedTrainLocal-v0/pickup",
    "BabyAI-MixedTrainLocal-v0/open",
    "BabyAI-MixedTrainLocal-v0/putnext",
    "BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to",
]


def make_babyai_env(env_name, task, config, render_mode: Optional[str] = None):
    """
    Build a BabyAI ``gym.Env`` and wrap it with ``BabyAITextCleanLangWrapper``.

    Parameters
    ----------
    env_name:
        Currently unused; kept for backward-compatible call sites.
    task:
        Either ``"BabyAI-MixedTrainLocal-v0/<goal>"`` (resample until mission kind matches)
        or a **plain Gym id** such as ``"BabyAI-GoToLocal-v0"`` (single ``gym.make``).

    config:
        Must expose ``config.envs.babyai_kwargs``: keyword arguments forwarded to both
        ``gym.make`` and ``BabyAITextCleanLangWrapper``.

    render_mode:
        Optional render mode passed through to ``gym.make`` (e.g. ``"rgb_array"``).
    """
    if task.startswith("BabyAI-MixedTrainLocal-v0/"):
        # Expects exactly one "/" so unpack yields (base_task, goal), e.g.
        # "BabyAI-MixedTrainLocal-v0/goto" -> base "BabyAI-MixedTrainLocal-v0", goal "goto".
        base_task, goal = task.split("/")
        # MixedTrain draws a random mission family each time we construct the env; loop
        # until the sampled mission matches the requested ``goal`` tag.
        while 1:
            env = gym.make(base_task, render_mode=render_mode, **config.envs.babyai_kwargs)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                break
    else:
        env = gym.make(task, render_mode=render_mode, **config.envs.babyai_kwargs)

    # Wrap the underlying env: discrete actions in, string actions and text obs out.
    env = BabyAITextCleanLangWrapper(env, **config.envs.babyai_kwargs)

    return env
