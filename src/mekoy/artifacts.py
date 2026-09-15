"""Where a downloadable bundle is written.

PLAN §41.17 asks for object storage. A hosted bucket needs an account, so what
ships is the half that does not: a tiny store protocol with a filesystem backend,
which is what the local control plane needs and what the tests can exercise.

The protocol is the point. `LocalArtifacts` and an S3 bucket differ only in where
the bytes land, so a hosted backend is a second implementation of three methods
rather than a change to the deploy route. Nothing here pretends a bucket exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from mekoy.bundle import write_bundle
from mekoy.dataset import TaskExample
from mekoy.errors import CompileError
from mekoy.spec import SystemSpec

__all__ = [
    "ARTIFACTS_DIR_ENV",
    "DEFAULT_ROOT",
    "ArtifactStore",
    "LocalArtifacts",
    "store_from_env",
    "write_bundle_to",
]

ARTIFACTS_DIR_ENV = "MEKOY_ARTIFACTS_DIR"
#: Beside the process, which is right for a laptop and wrong for a container
#: without a volume. The env var is how a deployment says so.
DEFAULT_ROOT = Path("artifacts")


@runtime_checkable
class ArtifactStore(Protocol):
    """Somewhere a bundle can be written and located."""

    def locator(self, system_id: str) -> str:
        """A human-usable location for this System's bundle."""
        ...

    def root(self) -> Path:
        """Local path to write into. A hosted backend materialises it first."""
        ...


@dataclass(frozen=True, slots=True)
class LocalArtifacts:
    """Filesystem-backed store."""

    base: Path = DEFAULT_ROOT

    def root(self) -> Path:
        """Create and return the directory bundles are written under."""
        self.base.mkdir(parents=True, exist_ok=True)
        return self.base

    def locator(self, system_id: str) -> str:
        """The bundle directory for one System."""
        return str(self.base / system_id)


def store_from_env() -> ArtifactStore:
    """Read the artifact root from the environment."""
    root = os.environ.get(ARTIFACTS_DIR_ENV, "").strip()
    return LocalArtifacts(base=Path(root) if root else DEFAULT_ROOT)


def write_bundle_to(
    store: ArtifactStore,
    system_id: str,
    spec: SystemSpec,
    report_text: str,
    *,
    examples: tuple[TaskExample, ...] = (),
) -> Path:
    """Write a bundle through the store, guarding the path it resolves to."""
    target = Path(store.locator(system_id)).resolve()
    root = store.root().resolve()
    if root not in target.parents:
        msg = f"refusing to write a bundle outside {root}: {target}"
        raise CompileError(message=msg)
    return write_bundle(target, spec, report_text, examples=examples)
