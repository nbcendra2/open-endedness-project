"""Functionality: Google Gemini with JSON output constrained by response_schema

Accepts the same OpenAI-style message list (role and content) and converts it
to Gemini user/model turns plus optional system_instruction.

Includes retry with backoff, truncated-JSON repair, and context windowing
to stay within model limits on long games (L10+).
"""

import json
import logging
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

from llm_clients.base_client import BaseLLMClient
from llm_clients.json_utils import ensure_action_in_valid, parse_maybe_markdown_json

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Rough character budget for the *contents* list (not counting system prompt).
# Gemini 2.5 Flash Lite supports ~130 k input tokens; we stay well under.
_MAX_CONTENT_CHARS = 80_000

# Output-token budgets per call type (generous enough to avoid truncation)
_OUTPUT_TOKENS_ACTION = 512
_OUTPUT_TOKENS_PLAN = 1024
_OUTPUT_TOKENS_REFLECT = 2048
_OUTPUT_TOKENS_FREE = 2048

# ---------------------------------------------------------------------------
# Retry / repair helpers
# ---------------------------------------------------------------------------

_TRANSIENT_SUBSTRINGS = (
    "timeout", "timed out", "deadline", "500", "502", "503", "504",
    "resource exhausted", "rate limit", "overloaded",
    "unterminated string", "expecting value", "json",
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks transient or is a JSON parse error."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    msg = str(exc).lower()
    return any(sub in msg for sub in _TRANSIENT_SUBSTRINGS)


def _repair_truncated_json(raw: str) -> dict | None:
    """Best-effort repair for JSON that was cut off mid-string."""
    raw = raw.strip()
    if not raw:
        return None

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    # Try as-is first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Close any open strings then close braces/brackets
    repaired = raw
    in_string = False
    prev = ""
    for ch in repaired:
        if ch == '"' and prev != '\\':
            in_string = not in_string
        prev = ch
    if in_string:
        repaired += '"'

    opens_brace = repaired.count('{') - repaired.count('}')
    for _ in range(max(opens_brace, 0)):
        repaired += '}'
    opens_bracket = repaired.count('[') - repaired.count(']')
    for _ in range(max(opens_bracket, 0)):
        repaired += ']'

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: find outermost { and force-close
    brace = raw.find('{')
    if brace != -1:
        fragment = raw[brace:].rstrip()
        if not fragment.endswith('}'):
            fragment += '"}'
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass

    return None


def _retry_generate(fn, *, max_retries: int = 3, base_delay: float = 1.0):
    """Call *fn* with exponential back-off on retryable failures."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Gemini attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                attempt + 1, max_retries + 1, type(exc).__name__, exc, delay,
            )
            time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Context windowing
# ---------------------------------------------------------------------------

def _window_contents(contents: list[dict], max_chars: int = _MAX_CONTENT_CHARS) -> list[dict]:
    """Keep the first message (initial observation) and as many recent
    messages as fit within *max_chars*.  This prevents L10 trajectories
    from blowing out the context window while preserving the game setup
    and the most recent observations which matter most for decision-making.
    """
    if not contents:
        return contents

    total = sum(len(str(m.get("parts", ""))) for m in contents)
    if total <= max_chars:
        return contents

    # Always keep the first message (game intro / initial observation)
    first = contents[0]
    first_len = len(str(first.get("parts", "")))
    budget = max_chars - first_len

    # Pack from the end
    kept_tail: list[dict] = []
    running = 0
    for msg in reversed(contents[1:]):
        msg_len = len(str(msg.get("parts", "")))
        if running + msg_len > budget:
            break
        kept_tail.append(msg)
        running += msg_len
    kept_tail.reverse()

    windowed = [first] + kept_tail
    dropped = len(contents) - len(windowed)
    if dropped > 0:
        logger.info(
            "Context windowed: kept %d/%d messages (dropped %d oldest)",
            len(windowed), len(contents), dropped,
        )
    return windowed


class GeminiClient(BaseLLMClient):
    """Google Gemini provider: native structured-output (response_schema)"""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model_name = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)

    @staticmethod
    def _convert_messages(messages: list):
        """Split OpenAI-style messages into (system_instruction, contents).

        Gemini uses 'user' / 'model' roles (not 'assistant').
        """
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_instruction = text
            elif role == "user":
                contents.append({"role": "user", "parts": [text]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
        return system_instruction, contents

    def _make_model(self, system_instruction: str | None):
        """Create a GenerativeModel with the given system instruction."""
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
        )

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    def generate(self, messages, temperature=0.2, timeout=30):
        system_instruction, contents = self._convert_messages(messages)
        contents = _window_contents(contents)
        model = self._make_model(system_instruction)

        def _call():
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=_OUTPUT_TOKENS_FREE,
                ),
                request_options={"timeout": timeout},
            )
            return response.text

        return _retry_generate(_call)

    def generate_action_structured(self, messages, valid_actions, temperature=0.2, timeout=30):
        system_instruction, contents = self._convert_messages(messages)
        contents = _window_contents(contents)
        model = self._make_model(system_instruction)

        schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "action": {"type": "string", "enum": valid_actions},
            },
            "required": ["reason", "action"],
        }

        def _call():
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=_OUTPUT_TOKENS_ACTION,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
                request_options={"timeout": timeout},
            )
            raw = response.text
            try:
                result = parse_maybe_markdown_json(raw)
            except json.JSONDecodeError:
                result = _repair_truncated_json(raw)
                if result is None:
                    raise
                logger.warning("Repaired truncated Gemini JSON: %s", result)
            return ensure_action_in_valid(result, valid_actions)

        return _retry_generate(_call)

    def generate_planning_structured(self, messages, temperature=0.3, timeout=30):
        system_instruction, contents = self._convert_messages(messages)
        contents = _window_contents(contents)
        model = self._make_model(system_instruction)

        schema = {
            "type": "object",
            "properties": {
                "plan": {"type": "string"},
            },
            "required": ["plan"],
        }

        def _call():
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=_OUTPUT_TOKENS_PLAN,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
                request_options={"timeout": timeout},
            )
            raw = response.text
            try:
                return parse_maybe_markdown_json(raw)
            except json.JSONDecodeError:
                result = _repair_truncated_json(raw)
                if result is None:
                    raise
                logger.warning("Repaired truncated Gemini JSON: %s", result)
                return result

        return _retry_generate(_call)

    def generate_reflection(self, messages, temperature=0.3, timeout=60):
        system_instruction, contents = self._convert_messages(messages)
        contents = _window_contents(contents)
        model = self._make_model(system_instruction)

        schema = {
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
        }

        def _call():
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=_OUTPUT_TOKENS_REFLECT,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
                request_options={"timeout": timeout},
            )
            raw = response.text
            try:
                return parse_maybe_markdown_json(raw)
            except json.JSONDecodeError:
                result = _repair_truncated_json(raw)
                if result is None:
                    raise
                logger.warning("Repaired truncated Gemini JSON: %s", result)
                return result

        return _retry_generate(_call)