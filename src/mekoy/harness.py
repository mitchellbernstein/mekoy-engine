"""Schema extract → verify → retry.

The loop is model + deterministic policy. Verifier errors are the next step's
context. We do not resend the whole previous prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import assert_never

from pydantic import BaseModel

from mekoy.runtime import Completer
from mekoy.tasks import RESTAURANT, Task
from mekoy.verify import (
    VerifyFail,
    VerifyOk,
    VerifyResult,
    explain,
    parse_and_gate,
)

#: Instruction variants live with the task, not with the loop.
PROMPTS: dict[str, str] = RESTAURANT.prompt_variants
DEFAULT_PROMPT = "default"


@dataclass(frozen=True, slots=True)
class Decode:
    """How the model is asked to answer: instructions plus format enforcement."""

    system: str = RESTAURANT.prompt
    constrained: bool = True
    temperature: float = 0.0
    #: When set, the server is asked to constrain generation to this schema.
    schema: dict[str, object] | None = None


def extract(
    completer: Completer,
    *,
    text: str,
    shots: tuple[tuple[str, object], ...] = (),
    retries: int = 1,
    decode: Decode | None = None,
    task: Task = RESTAURANT,
) -> VerifyResult:
    """Model call, then the task's gate. Retry with check failures only."""
    how = decode or Decode(system=task.prompt)
    attempts = retries + 1
    user = _user_prompt(text, shots)
    result: VerifyResult = VerifyFail(reasons=("no model call",))
    for attempt in range(attempts):
        raw = completer.complete(
            system=how.system,
            user=user,
            constrained=how.constrained,
            temperature=how.temperature,
            schema=how.schema,
        )
        result = parse_and_gate(_strip_fence(raw), model=task.model, gate=task.gate)
        match result:
            case VerifyOk():
                return result
            case VerifyFail():
                if attempt + 1 >= attempts:
                    return result
                user = _repair_prompt(text, shots, result)
            case _ as unreachable:
                assert_never(unreachable)
    return result


def _user_prompt(text: str, shots: tuple[tuple[str, object], ...]) -> str:
    parts: list[str] = []
    for shot_text, gold in shots:
        # Any task's schema: shots are dumped, not introspected.
        dumped = (
            gold.model_dump_json() if isinstance(gold, BaseModel) else json.dumps(gold)
        )
        parts.append(f"Example text:\n{shot_text}\nExample JSON:\n{dumped}")
    parts.append(f"Text:\n{text}\nJSON:")
    return "\n\n".join(parts)


def _repair_prompt(
    text: str,
    shots: tuple[tuple[str, object], ...],
    failed: VerifyFail,
) -> str:
    """Next-step context: shots, the document, verifier reasons. Not the last prompt."""
    base = _user_prompt(text, shots)
    return (
        f"{base}\n\nDeterministic checks failed: {explain(failed)}\n"
        "Return corrected JSON only. Do not explain."
    )


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
