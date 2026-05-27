from typing import Any

import anthropic
from pydantic import ValidationError

from agent.llm.prompt import SYSTEM
from agent.llm.schema import Decision, hold


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in response")
    return text[start : end + 1]


class Decider:
    client: Any

    def __init__(self, api_key: str, model: str, client: Any = None) -> None:
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=api_key)

    def decide(self, user_prompt: str) -> tuple[Decision, str]:
        last_error = "unknown error"
        for _ in range(2):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                raw = _extract_json(text)
                return Decision.model_validate_json(raw), raw
            except (ValueError, ValidationError, IndexError, KeyError) as exc:
                last_error = f"parse error: {exc}"
            except Exception as exc:
                last_error = f"api error: {exc}"
        fallback = hold(f"LLM decision failed: {last_error}")
        return fallback, fallback.model_dump_json()
