"""Functionality: Shared JSON parsing and small validation helpers for LLM replies

Used when models may wrap JSON in markdown fences or return an invalid action enum
"""

import json


def parse_maybe_markdown_json(text: str) -> dict:
    """Parse JSON from model text, stripping optional markdown code fences"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def ensure_action_in_valid(result: dict, valid_actions: list[str]) -> dict:
    """If action is missing or not in valid_actions, set it to valid_actions[0]"""
    if result.get("action") not in valid_actions:
        result["action"] = valid_actions[0]
    return result
