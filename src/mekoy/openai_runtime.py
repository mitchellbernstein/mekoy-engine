"""GPT-6 Astra as a bake-off baseline. Never as training data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mekoy.errors import CompileError, ModelUnreachableError
from mekoy.http_client import create_client

ASTRA_MODEL = "gpt-6-astra"
_INPUT_USD_PER_M = 10.0
_OUTPUT_USD_PER_M = 50.0
_HTTP_ERROR_MIN = 400


@dataclass(slots=True)
class TokenMeter:
    """Accumulates chat-completion usage. Mutable by design."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def usd(self) -> float:
        """List price from OpenAI's GPT-6 Astra card ($10 / $50 per 1M)."""
        return (
            self.input_tokens / 1_000_000 * _INPUT_USD_PER_M
            + self.output_tokens / 1_000_000 * _OUTPUT_USD_PER_M
        )


class _Usage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _ChoiceMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    content: str | None = None


class _Choice(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    message: _ChoiceMessage


class _ChatResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    choices: tuple[_Choice, ...] = Field(min_length=1)
    usage: _Usage | None = None


class OpenAICompleter:
    """OpenAI chat completions. Opt-in baseline only. Never write outputs to SFT."""

    #: Closed API. Legal as a bake-off baseline, never as compile data.
    local: ClassVar[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str = ASTRA_MODEL,
        meter: TokenMeter | None = None,
    ) -> None:
        """Store the key in memory. Never log it."""
        if not api_key:
            msg = "OPENAI_API_KEY is missing"
            raise CompileError(message=msg)
        self._api_key: str = api_key
        self._model: str = model
        self.meter: TokenMeter = meter if meter is not None else TokenMeter()

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        """One JSON chat completion. No outputs are stored for training."""
        url = "https://api.openai.com/v1/chat/completions"
        body: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": temperature,
            "max_completion_tokens": 2048,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "outcome",
                    "schema": schema,
                    "strict": True,
                },
            }
        elif constrained:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            with create_client(headers=headers) as client:
                response = client.post(url, json=body)
                status = getattr(response, "status_code", 200)
                if status >= _HTTP_ERROR_MIN:
                    body = getattr(response, "text", "")[:500]
                    msg = f"OpenAI HTTP {status}: {body}"
                    raise ModelUnreachableError(message=msg)
                payload = _ChatResponse.model_validate(response.json())
        except httpx2.HTTPError as exc:
            msg = f"OpenAI unreachable: {exc}"
            raise ModelUnreachableError(message=msg) from exc
        except ValidationError as exc:
            msg = f"OpenAI returned an unexpected body: {exc}"
            raise ModelUnreachableError(message=msg) from exc
        if payload.usage is not None:
            self.meter.input_tokens += payload.usage.prompt_tokens
            self.meter.output_tokens += payload.usage.completion_tokens
        content = payload.choices[0].message.content
        if content is None or not content.strip():
            msg = "OpenAI returned empty content"
            raise ModelUnreachableError(message=msg)
        return content


def api_key_from_env() -> str:
    """Read OPENAI_API_KEY. Empty string if unset."""
    return os.environ.get("OPENAI_API_KEY", "").strip()
