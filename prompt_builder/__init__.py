"""Functionality: Package entry for prompt construction (trajectory to chat messages)

The folder name prompt_builder refers to assembling LLM prompts; history.py
holds HistoryPromptBuilder because it builds prompts from step-by-step history
"""

from .history import HistoryPromptBuilder

__all__ = ["HistoryPromptBuilder"]
