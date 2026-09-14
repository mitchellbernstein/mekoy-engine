"""Frozen portable System spec. JSON on disk, types in process."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from mekoy.errors import CompileError

_FROZEN: ConfigDict = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Slos(BaseModel):
    """Quality, dollars per document, and latency gates."""

    model_config: ClassVar[ConfigDict] = _FROZEN

    quality: float = Field(ge=0.0, le=1.0)
    cost_per_doc: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)


class OwnershipFlags(BaseModel):
    """Whether the System is owned at runtime and can leave our hosts."""

    model_config: ClassVar[ConfigDict] = _FROZEN

    runtime_owned: bool
    downloadable: bool


class SystemSpec(BaseModel):
    """Task, schema, SLOs, winner model, harness knobs, ownership."""

    model_config: ClassVar[ConfigDict] = _FROZEN

    task: str = Field(min_length=1)
    json_schema: dict[str, object] = Field(
        min_length=1,
        validation_alias=AliasChoices("schema", "json_schema"),
        serialization_alias="schema",
    )
    slos: Slos
    model_id: str = Field(min_length=1)
    k_shot: int = Field(ge=0)
    retries: int = Field(ge=0)
    ownership: OwnershipFlags


def write_spec(path: Path, spec: SystemSpec) -> Path:
    """Write spec.json. Parent directories must already exist."""
    payload = spec.model_dump_json(indent=2, by_alias=True)
    _ = path.write_text(payload + "\n", encoding="utf-8")
    return path


def load_spec(path: Path) -> SystemSpec:
    """Parse spec.json from a file or a System directory."""
    target = path / "spec.json" if path.is_dir() else path
    if not target.is_file():
        msg = f"spec not found: {target}"
        raise CompileError(message=msg)
    try:
        return SystemSpec.model_validate_json(target.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        msg = f"{target}: {exc}"
        raise CompileError(message=msg) from exc
