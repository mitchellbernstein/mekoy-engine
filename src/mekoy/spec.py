"""Frozen portable System spec. JSON on disk, types in process."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from mekoy.errors import CompileError
from mekoy.harness import DEFAULT_PROMPT

if TYPE_CHECKING:
    from mekoy.search import HarnessConfig

_FROZEN: ConfigDict = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

#: The current spec format. One constant, because three places need to agree on it
#: and they drifted the moment a bundle wrote a version its own reader rejected.
SPEC_VERSION = 2

#: The version from which a bundle records the full harness. Below it, the harness axes
#: are silently defaulted and a verifier must say so rather than score a guess.
_HARNESS_RECORDED_FROM = 2


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
    """Task, schema, SLOs, winner model, harness knobs, ownership.

    `spec_version` exists because a downloaded System outlives the engine that wrote it.
    Without it, a bundle written by one format and read by another cannot detect that it
    is being misread: it would parse the fields it recognises and quietly default the
    rest, which is the one failure a portable artifact must never have.
    """

    model_config: ClassVar[ConfigDict] = _FROZEN

    #: Bumped when the meaning of a field changes. A reader must refuse a newer one.
    spec_version: int = Field(default=1, ge=1)
    task: str = Field(min_length=1)
    #: What the base model may be used for. PLAN §24a requires a listing to carry it:
    #: some weights cannot be sold as weights at all, and a caller cannot know what they
    #: may do with a result without it. Empty means unknown, which the review refuses.
    license: str = ""
    #: Where the labeled examples came from. PLAN §24a: no publication when this cannot
    #: be shown to be licensed. Empty means unattested, which is also a refusal.
    data_source: str = ""
    json_schema: dict[str, object] = Field(
        min_length=1,
        validation_alias=AliasChoices("schema", "json_schema"),
        serialization_alias="schema",
    )
    slos: Slos
    model_id: str = Field(min_length=1)
    k_shot: int = Field(ge=0)
    retries: int = Field(ge=0)
    #: The rest of the harness. These were missing, and their absence was the quietest
    #: bug in the product: a bundle recorded `k_shot` and `retries` and dropped
    #: the other
    #: five axes, so a recipient rebuilt a *different* harness and got a different score
    #: than the one they were sold. The restaurant winner is `k=4 r=0 schema strict`; a
    #: spec without `prompt` hands over the default brief instead of the strict one, and
    #: nothing anywhere said so.
    #:
    #: Defaults are the harness the engine uses when nothing else is chosen, so a spec
    #: written before these fields existed still loads, and `carries_harness` reports
    #: whether it actually does rather than letting a reader assume it does.
    constrained: bool = True
    prompt: str = DEFAULT_PROMPT
    #: Whether the decode was pinned to the JSON schema - a separate axis from
    #: `constrained`, because a constrained decode of the wrong shape still returns
    #: something. Named `schema_constrained` rather than `schema` because the JSON key
    #: `schema` already carries the schema itself, and shadowing it would silently
    #: overwrite one of the two.
    schema_constrained: bool = False
    consistency: int = Field(default=1, ge=1)
    bootstrap: bool = False
    ownership: OwnershipFlags

    @property
    def carries_harness(self) -> bool:
        """Whether this spec records enough to rebuild the harness that was measured.

        A spec written before the harness axes existed has `spec_version` 1 and silent
        defaults for all five, which is indistinguishable from a System that genuinely
        chose those defaults. Rather than guess, a verifier asks this and reports that
        it cannot check - a wrong score presented as a right one is worse than none.
        """
        return self.spec_version >= _HARNESS_RECORDED_FROM

    def harness(self) -> HarnessConfig:
        """The harness this spec describes, as the search understands it.

        Imported here rather than at module scope because `search` reads `Slos` out of
        this module, so a top-level import of `HarnessConfig` would be a cycle.
        """
        from mekoy.search import HarnessConfig  # noqa: PLC0415 - see docstring

        return HarnessConfig(
            k_shot=self.k_shot,
            retries=self.retries,
            constrained=self.constrained,
            prompt=self.prompt,
            schema=self.schema_constrained,
            consistency=self.consistency,
            bootstrap=self.bootstrap,
            model=self.model_id,
        )


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
