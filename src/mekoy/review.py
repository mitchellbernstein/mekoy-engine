"""Whether a System may be published, and what a reviewer can prove about it.

PLAN §24a names the review steps: license of the base model, training-data
attestation, a harness sandbox check, our own eval, and a policy scan. This module
decides which of them can be settled from the artifact, and refuses publication when
one cannot.

The rule the whole file exists to enforce: **nothing ranks on a number a third party
cannot recompute.** A score we assert and nobody can check is not evidence, and a
catalog ranked by one would be selling evidence we do not have. A System whose score
cannot be recomputed from its own bundle is *unverifiable*, and an unverifiable System
is not listable - regardless of how good the number looks.

Every check returns a finding rather than a boolean, because "refused" without a reason
is indistinguishable from a bug, and because the publisher needs to know which of the
five things to go and fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from mekoy.spec import SystemSpec

__all__ = [
    "Check",
    "Decision",
    "Finding",
    "OptIn",
    "Review",
    "review",
]


class Check(StrEnum):
    """The review steps from PLAN §24a, in the order a reviewer would run them."""

    OPT_IN = "opt_in"
    LICENSE = "license"
    ATTESTATION = "attestation"
    VERIFIABLE = "verifiable"
    HARNESS = "harness"


class Decision(StrEnum):
    """What the gate concluded."""

    LISTABLE = "listable"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class Finding:
    """One review step's outcome, with the reason a publisher needs to act on."""

    check: Check
    #: Keyword-only so a bare `True` at a call site cannot be read as anything else.
    passed: bool = field(kw_only=True)
    detail: str = ""

    @property
    def blocked(self) -> bool:
        """True when this alone stops publication."""
        return not self.passed


@dataclass(frozen=True, slots=True)
class OptIn:
    """The two confirmations PLAN §24a requires.

    Two separate flags rather than one, because the point of asking twice is that the
    first answer is often "yes, obviously" and the second is where someone reads what
    they are agreeing to. A single boolean cannot tell those apart, and a publisher who
    clicked through once has not consented.
    """

    consented: bool = False
    #: The publisher has been shown that their examples become public.
    acknowledged_data_becomes_public: bool = False

    @property
    def complete(self) -> bool:
        """Both confirmations given."""
        return self.consented and self.acknowledged_data_becomes_public


@dataclass(frozen=True, slots=True)
class Review:
    """The findings, and what they add up to."""

    findings: tuple[Finding, ...]
    decision: Decision

    @property
    def listable(self) -> bool:
        """Whether the catalog may carry this System."""
        return self.decision is Decision.LISTABLE

    @property
    def blockers(self) -> tuple[Finding, ...]:
        """Only the findings that stopped it."""
        return tuple(f for f in self.findings if f.blocked)

    def explain(self) -> str:
        """One line a publisher can act on, or a clean bill."""
        if self.listable:
            return "listable: all review checks passed."
        reasons = "; ".join(f"{f.check.value}: {f.detail}" for f in self.blockers)
        return f"refused: {reasons}"


def review(
    spec: SystemSpec,
    *,
    opt_in: OptIn,
    bundle_dir: Path | None = None,
    score_is_recomputable: bool = False,
    harness_tools: tuple[str, ...] = (),
) -> Review:
    """Run every review step and decide.

    `score_is_recomputable` is passed in rather than computed here, because the
    thing that answers it is `verify-bundle`, which lives with the evaluator. This
    module's job is to refuse when the answer is no, not to be the answer.

    Everything else is read from the spec, so a reviewer and a bundle cannot disagree
    about what was attested.
    """
    findings = (
        _check_opt_in(opt_in),
        _check_license(spec),
        _check_attestation(spec),
        _check_verifiable(score_is_recomputable, bundle_dir),
        _check_harness(harness_tools),
    )
    decision = (
        Decision.LISTABLE if all(f.passed for f in findings) else Decision.REFUSED
    )
    return Review(findings=findings, decision=decision)


def _check_opt_in(opt_in: OptIn) -> Finding:
    if opt_in.complete:
        return Finding(Check.OPT_IN, passed=True, detail="both confirmations given")
    missing = []
    if not opt_in.consented:
        missing.append("consent")
    if not opt_in.acknowledged_data_becomes_public:
        missing.append("the data-becomes-public acknowledgement")
    return Finding(Check.OPT_IN, passed=False, detail=f"missing {', '.join(missing)}")


def _check_license(spec: SystemSpec) -> Finding:
    """A listing must carry what its base model is licensed under.

    PLAN §24a: some weights cannot be sold as weights at all, so a listing without a
    licence cannot be routed to honestly - the caller has no way to know what they are
    allowed to do with the result.
    """
    if not spec.license.strip():
        return Finding(
            Check.LICENSE,
            passed=False,
            detail="no licence, so the base model's terms are unknown",
        )
    return Finding(
        Check.LICENSE, passed=True, detail=f"base model licence: {spec.license}"
    )


def _check_attestation(spec: SystemSpec) -> Finding:
    """Where the training data came from has to be stated, not assumed.

    PLAN §24a is explicit: no publication of a System whose training data cannot be
    shown to be licensed. An empty attestation is a refusal, never a pass.

    Read from the spec rather than taken as an argument, so there is one source of
    truth. When it was an argument, the tests passed a value the spec did not carry and
    the check contradicted itself.
    """
    stated = spec.data_source.strip()
    if not stated:
        return Finding(
            Check.ATTESTATION,
            passed=False,
            detail="no statement of where the training data came from",
        )
    return Finding(
        Check.ATTESTATION,
        passed=True,
        detail=f"data attested as: {stated}",
    )


def _check_verifiable(score_is_recomputable: bool, bundle_dir: Path | None) -> Finding:
    """The check that decides whether this catalog can exist at all.

    Our published number is 0.940 on restaurant calls. Nobody outside this repository
    can reproduce it: the corpus is not shipped and there is no command that re-scores a
    bundle. A catalog ranking Systems by such a number would assert evidence it does not
    have, and a caller could not tell a real 0.94 from a lucky run.

    So this refuses, and the refusal is the honest state of Phase III until
    `verify-bundle` exists and the corpora it needs are published.
    """
    if not score_is_recomputable:
        return Finding(
            Check.VERIFIABLE,
            passed=False,
            detail="the advertised score cannot be recomputed from the bundle: no "
            "`verify-bundle` command and no published held-out corpus, so ranking "
            "on it would assert evidence that does not exist",
        )
    if bundle_dir is None or not bundle_dir.is_dir():
        return Finding(
            Check.VERIFIABLE,
            passed=False,
            detail="no bundle directory to re-score",
        )
    return Finding(
        Check.VERIFIABLE,
        passed=True,
        detail="the score can be recomputed from the bundle",
    )


def _check_harness(harness_tools: tuple[str, ...]) -> Finding:
    """A System's harness must not reach out without the caller knowing.

    PLAN §24a asks for a sandbox check because a harness that opens a socket or a file
    is doing something the caller did not ask for. This repository's v1 harness is a
    schema, a decode, a verify, and a retry loop - it has no tools - so the expected
    answer is an empty set, and anything present is reported rather than allowed.
    """
    if not harness_tools:
        return Finding(Check.HARNESS, passed=True, detail="the harness calls no tools")
    return Finding(
        Check.HARNESS,
        passed=False,
        detail=f"uncatalogued harness tools: {sorted(harness_tools)}",
    )
