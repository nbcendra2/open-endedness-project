"""Functionality: OpenAI Chat Completions with native structured JSON via response_format

Builds OpenAI-specific response_format wrappers inline; the API can enforce strict
JSON schema including action enums where supported
"""

import os
import json
import openai
from dotenv import load_dotenv
from typing import Dict, List

from llm_clients.base_client import BaseLLMClient

load_dotenv()


def _utf8_safe_text(value) -> str:
    """Ensure text is a plain str safe for JSON/HTTP UTF-8 (no lone surrogates)."""
    if value is None:
        return ""
    s = str(value)
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _normalize_messages(messages) -> List[Dict]:
    out: List[Dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and "text" in p:
                    parts.append(_utf8_safe_text(p.get("text")))
                else:
                    parts.append(_utf8_safe_text(p))
            text = "\n".join(parts)
        else:
            text = _utf8_safe_text(content)
        out.append({"role": role, "content": text})
    return out


def _normalize_action_enum(valid_actions: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for a in valid_actions or []:
        s = _utf8_safe_text(a).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _action_response_format(valid_actions: List[str]) -> Dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "action": {"type": "string", "enum": valid_actions},
                },
                "required": ["reason", "action"],
                "additionalProperties": False,
            },
        },
    }


def _planning_response_format() -> Dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
        },
    }


def _reflection_response_format() -> Dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "episode_reflection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "strategy": {"type": "string"},
                    "lessons": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["summary", "strategy", "lessons"],
                "additionalProperties": False,
            },
        },
    }


class OpenAIClient(BaseLLMClient):
    """OpenAI provider: native structured-output (response_format)"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, messages, temperature=0.2, timeout=10):
        # Plain chat completion; unstructured text in the reply
        msgs = _normalize_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            timeout=timeout,
        )
        return response.choices[0].message.content

    def generate_action_structured(self, messages, valid_actions, temperature=0.2, timeout=10):
        msgs = _normalize_messages(messages)
        actions = _normalize_action_enum(valid_actions)
        if not actions:
            raise ValueError(
                "valid_actions is empty after normalization; cannot build structured action schema"
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            timeout=timeout,
            response_format=_action_response_format(actions),
        )
        return json.loads(response.choices[0].message.content)

    def generate_planning_structured(self, messages, temperature=0.3, timeout=15):
        msgs = _normalize_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            timeout=timeout,
            response_format=_planning_response_format(),
        )
        return json.loads(response.choices[0].message.content)

    def generate_reflection(self, messages, temperature=0.3, timeout=30):
        msgs = _normalize_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            timeout=timeout,
            response_format=_reflection_response_format(),
        )
        return json.loads(response.choices[0].message.content)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=_utf8_safe_text(text),
        )
        return response.data[0].embedding
