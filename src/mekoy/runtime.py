"""OpenAI-compatible local runtime (Ollama by default)."""

from __future__ import annotations

from typing import ClassVar, Protocol

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mekoy.errors import ModelUnreachableError
from mekoy.http_client import create_client


class _ChoiceMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    content: str


class _Choice(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    message: _ChoiceMessage


class _ChatResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    choices: tuple[_Choice, ...] = Field(min_length=1)


class Completer(Protocol):
    """Anything that turns a system+user prompt into text."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        """Return model text for a system and user prompt.

        `schema` asks the server to constrain generation to that JSON schema. Not
        every OpenAI-compatible server enforces it — Ollama accepts the request and
        can return `{}` — so it is a search axis, not a default.
        """
        ...


class OllamaCompleter:
    """Chat completions against a local OpenAI-compatible server."""

    #: Runs on hardware we control, so its output is usable as compile data.
    local: ClassVar[bool] = True

    def __init__(self, *, base_url: str, model: str) -> None:
        """Point at an OpenAI-compatible /v1 server."""
        self._base_url: str = base_url.rstrip("/")
        self._model: str = model

    def complete(
        self,
        *,
        system: str,
        user: str,
        constrained: bool = True,
        temperature: float = 0.0,
        schema: dict[str, object] | None = None,
    ) -> str:
        """POST /chat/completions and return the assistant content."""
        url = f"{self._base_url}/chat/completions"
        body: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "outcome", "schema": schema},
            }
        elif constrained:
            body["response_format"] = {"type": "json_object"}
        try:
            with create_client() as client:
                response = client.post(url, json=body)
                _ = response.raise_for_status()
                payload = _ChatResponse.model_validate(response.json())
        except httpx2.HTTPError as exc:
            msg = f"model server unreachable at {url}: {exc}"
            raise ModelUnreachableError(message=msg) from exc
        except ValidationError as exc:
            msg = f"model returned an unexpected body: {exc}"
            raise ModelUnreachableError(message=msg) from exc
        content = payload.choices[0].message.content
        if not content.strip():
            msg = "model returned empty content"
            raise ModelUnreachableError(message=msg)
        return content
