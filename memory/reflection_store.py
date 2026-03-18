import json
import os
from typing import List


def _task_id_from_env_name(env_name: str) -> str:
    """Sanitize env_name to a filesystem-friendly task identifier."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(env_name))


def _reflections_path(task_id: str) -> str:
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", f"reflections_{task_id}.json")


def load_reflections(env_name: str, max_reflections: int = 3) -> List[str]:
    """Load up to max_reflections most recent reflections for a given environment name."""
    task_id = _task_id_from_env_name(env_name)
    path = _reflections_path(task_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [str(x) for x in data[-max_reflections:]]
    except Exception:
        # Fail silently and return no reflections if the file is malformed
        return []


def append_reflection(env_name: str, reflection_text: str, max_reflections: int = 3) -> None:
    """Append a new reflection for the given environment name, keeping only the last max_reflections."""
    task_id = _task_id_from_env_name(env_name)
    path = _reflections_path(task_id)

    reflections = load_reflections(env_name, max_reflections=max_reflections)
    reflections.append(str(reflection_text))
    reflections = reflections[-max_reflections:]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(reflections, f, ensure_ascii=False, indent=2)

