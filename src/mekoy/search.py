"""Budgeted harness search with SLO gates and a Pareto winner rule.

PLAN §15: staged ASHA with Pareto secondary (quality, $, latency), stopping on
the SLO. This is the local, CPU-sized form of it: a candidate pool over
{k-shot, retries, decode, prompt}, evaluated stage by stage, pruned on the hard
gate, ranked on the Pareto front, and stopped the moment an SLO-clearing
candidate appears.

Selection reads `dev` only. `test` is never touched here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from mekoy.dataset import ExampleRecord
from mekoy.errors import CompileError
from mekoy.harness import DEFAULT_PROMPT, PROMPTS, Decode, extract
from mekoy.outcome import RestaurantOutcome
from mekoy.score import ExampleScore
from mekoy.spec import Slos
from mekoy.tasks import RESTAURANT, Scored, Task
from mekoy.verify import VerifyFail, VerifyOk, VerifyResult, parse_and_verify

if TYPE_CHECKING:
    from mekoy.runtime import Completer

__all__ = [
    "Budget",
    "HarnessConfig",
    "SearchSpace",
    "Slos",
    "Trial",
    "evaluate",
    "pareto_front",
    "search",
]

_DEFAULT_TRIALS = 5
#: Sampling temperature for self-consistency. Zero would make every sample identical.
_CONSENSUS_TEMPERATURE = 0.7
#: Fields that decide the safety call, and therefore get voted on.
_CLOSED_FIELDS = ("intent", "status", "party_size", "booked")
#: Minibatch size for the first ASHA rung (PLAN §15.6: n=16-32).
_MINIBATCH = 16
#: Successive-halving prune factor (PLAN §15.7: ASHA eta).
_PRUNE_ETA = 2
#: Below this many dev rows, staging costs more than it saves.
_STAGING_FLOOR = 12
_PHRASE_FIELDS = ("restaurant", "when", "under_name")
#: Shots are the cheapest lever on quality, so the default space reaches higher
#: than the old cap of 4. PLAN §15.5 brackets shots at 0 and 4; a 60-row train
#: slice can afford more.
_MAX_K = 8
_K_LADDER = (0, 2, 4, 8)
_MIN_ROWS_FOR_PROMPT = 8


@dataclass(frozen=True, slots=True)
class Budget:
    """What the caller is willing to spend, and what counts as done.

    `trials` is a trial budget, not a metric-call budget: each candidate may
    spend many model calls. `slos` stops the search early once it is met.
    """

    trials: int = _DEFAULT_TRIALS
    seed: int = 0
    slos: Slos | None = None
    #: Successive halving on a minibatch before paying for a full dev pass.
    staged: bool = True


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """One point in the harness space: the knobs the compiler may turn."""

    k_shot: int = 0
    retries: int = 0
    constrained: bool = True
    prompt: str = DEFAULT_PROMPT
    consistency: int = 1
    #: Use bootstrapped demonstrations instead of the first k training rows.
    bootstrap: bool = False
    #: Ask the server to constrain generation to the task's JSON schema, rather
    #: than merely to valid JSON. Servers differ: llama.cpp enforces it, Ollama
    #: accepts it and can return `{}`, so this is a search axis and never a
    #: default.
    schema: bool = False
    #: Which base model to run. Empty means the completer the caller supplied.
    #: PLAN 15.5 samples over three open models, and 41.9 names `model` first in
    #: the controller's axes, so a search that cannot change the model is missing
    #: its largest lever.
    model: str = ""

    @property
    def label(self) -> str:
        """Short stable name for a report line."""
        decode = "grammar" if self.constrained else "free"
        if self.schema:
            decode = "schema"
        vote = f" vote{self.consistency}" if self.consistency > 1 else ""
        shots = " boot" if self.bootstrap else ""
        size = f"{self.model} " if self.model else ""
        return (
            f"{size}k={self.k_shot} r={self.retries} {decode} "
            f"{self.prompt}{vote}{shots}"
        )


@dataclass(frozen=True, slots=True)
class Trial:
    """One candidate config measured on one split."""

    config: HarnessConfig
    scores: tuple[Scored, ...]
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    #: Gate reasons seen on this candidate. Empty means nothing was rejected.
    reasons: tuple[str, ...] = ()

    @property
    def quality(self) -> float:
        """Mean per-doc field accuracy, free-text phrasing normalized."""
        if not self.scores:
            return 0.0
        return sum(s.quality for s in self.scores) / len(self.scores)

    @property
    def strict_quality(self) -> float:
        """Same, with raw string equality. Reported alongside quality."""
        if not self.scores:
            return 0.0
        return sum(s.strict_quality for s in self.scores) / len(self.scores)

    @property
    def schema_rate(self) -> float:
        """Fraction of docs that parsed and cleared the policy gate."""
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.schema_ok) / len(self.scores)

    @property
    def violations(self) -> int:
        """Docs rejected by the deterministic gate. Not a quality knob."""
        return sum(1 for s in self.scores if not s.schema_ok)

    def meets(self, slos: Slos) -> bool:
        """True when this trial clears every declared gate."""
        return (
            self.quality >= slos.quality
            and self.cost_usd <= slos.cost_per_doc
            and self.latency_ms <= slos.latency_ms
        )


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Cartesian product of the harness axes. Sampled, not fully enumerated."""

    k_shots: tuple[int, ...] = (0, 2, 4, 8)
    retries: tuple[int, ...] = (0, 1)
    constrained: tuple[bool, ...] = (True, False)
    prompts: tuple[str, ...] = (DEFAULT_PROMPT,)
    consistency: tuple[int, ...] = (1,)
    #: Off by default: bootstrapping costs a pass over the training set.
    bootstrap: tuple[bool, ...] = (False,)
    #: Both, because "always A/B unconstrained" applies to schema-constrained too.
    schema: tuple[bool, ...] = (False, True)
    #: Base models to try. Empty string means the caller's completer, so a search
    #: with one model behaves exactly as before.
    models: tuple[str, ...] = ("",)

    @classmethod
    def local(cls, *, train_n: int) -> SearchSpace:
        """Default local space. k-shot is capped by the shots actually available."""
        max_k = min(_MAX_K, train_n)
        k_shots = tuple(k for k in _K_LADDER if k <= max_k) or (0,)
        if k_shots[-1] != max_k:
            k_shots = (*k_shots, max_k)
        prompts = (
            (DEFAULT_PROMPT, "strict")
            if train_n >= _MIN_ROWS_FOR_PROMPT
            else (DEFAULT_PROMPT,)
        )
        return cls(
            k_shots=k_shots,
            retries=(0, 1),
            constrained=(True, False),
            prompts=prompts,
        )

    @classmethod
    def for_task(cls, task: Task, *, train_n: int) -> SearchSpace:
        """The default space, restricted to the variants this task actually has."""
        base = cls.local(train_n=train_n)
        prompts = tuple(task.prompt_variants) or (DEFAULT_PROMPT,)
        return cls(
            k_shots=base.k_shots,
            retries=task.retry_ladder,
            constrained=base.constrained,
            prompts=prompts,
            consistency=base.consistency,
            bootstrap=base.bootstrap,
            schema=base.schema,
            models=base.models,
        )

    @classmethod
    def single(cls) -> SearchSpace:
        """One candidate, for a smoke run on a laptop."""
        return cls(k_shots=(0,), retries=(1,), constrained=(True,), schema=(False,))

    def candidates(self) -> tuple[HarnessConfig, ...]:
        """Every point, deterministic order."""
        return tuple(
            HarnessConfig(
                k_shot=k,
                retries=r,
                constrained=c,
                prompt=p,
                consistency=v,
                bootstrap=b,
                schema=_schema,
                model=m,
            )
            for k in self.k_shots
            for r in self.retries
            for c in self.constrained
            for p in self.prompts
            for v in self.consistency
            for b in self.bootstrap
            for _schema in self.schema
            for m in self.models
        )

    def sample(self, n: int, *, seed: int = 0) -> tuple[HarnessConfig, ...]:
        """N candidates, or the whole space when it is smaller than N.

        Every prompt variant gets at least one trial before anything else is
        drawn, so a small budget still tests the axis that most often moves
        quality. Seeded, so a compile is reproducible: same seed, same winner.
        """
        pool = list(self.candidates())
        if n >= len(pool):
            return tuple(pool)
        rng = random.Random(seed)  # noqa: S311
        chosen: list[HarnessConfig] = []
        for prompt in self.prompts:
            if len(chosen) >= n:
                break
            group = [c for c in pool if c.prompt == prompt and c not in chosen]
            if group:
                chosen.append(rng.choice(group))
        rest = [c for c in pool if c not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: max(0, n - len(chosen))])
        return tuple(chosen)


def _system_for(config: HarnessConfig, task: Task) -> str:
    """Resolve a prompt variant for this task, falling back to its default.

    The task owns its prompts. This used to read a module-level table for the
    restaurant task, which meant any variant added at runtime — a reflected
    instruction, for instance — was silently replaced by the default and scored as
    the thing it was meant to improve on. A candidate that is never actually used
    looks exactly like a candidate that does not help.
    """
    if config.prompt in task.prompt_variants:
        return task.prompt_variants[config.prompt]
    return PROMPTS.get(config.prompt, task.prompt)


def _completer_for(
    config: HarnessConfig,
    default: Completer,
    models: Mapping[str, Completer] | None,
) -> Completer:
    """Pick the completer for this candidate's model.

    A missing entry is a configuration error, not a silent fallback: scoring a
    model nobody registered would attribute one model's numbers to another.
    """
    if not config.model:
        return default
    if models is None or config.model not in models:
        msg = (
            f"candidate asks for model {config.model!r} but no completer was "
            f"registered for it; known: {sorted(models or {})}"
        )
        raise CompileError(message=msg)
    return models[config.model]


def evaluate(  # noqa: PLR0913 - the evaluation entry point names its knobs
    completer: Completer,
    rows: tuple[ExampleRecord, ...],
    config: HarnessConfig,
    *,
    shots: tuple[ExampleRecord, ...] = (),
    task: Task = RESTAURANT,
    bootstrapped: tuple[tuple[str, object], ...] = (),
    models: Mapping[str, Completer] | None = None,
) -> Trial:
    """Run one candidate over `rows`, measuring quality, latency, and cost.

    With `config.bootstrap`, the demonstrations come from `bootstrapped` rather
    than the first k rows of `shots`: Stage 0 picks what to show, the k-shot axis
    only decides how much of it to show.
    """
    active = _completer_for(config, completer, models)
    meter = getattr(active, "meter", None)
    cost_before = getattr(meter, "usd", 0.0)
    system = _system_for(config, task)
    if config.bootstrap and bootstrapped:
        shot_pairs = bootstrapped[: config.k_shot]
    else:
        shot_pairs = tuple((row.text, row.outcome) for row in shots[: config.k_shot])
    # A schema is only meaningful when decoding is constrained at all.
    task_schema = (
        task.model.model_json_schema() if config.schema and config.constrained else None
    )
    scores: list[Scored] = []
    reasons: list[str] = []
    elapsed = 0.0
    for row in rows:
        start = perf_counter()
        # Self-consistency voting is defined over the restaurant's closed fields,
        # so other tasks take the single-sample path rather than being scored by
        # a voter that does not know their schema.
        if config.consistency > 1 and task.name == "restaurant":
            outcome = _voted_outcome(
                active,
                text=row.text,
                shots=shot_pairs,
                config=config,
                system=system,
                task=task,
            )
        else:
            outcome = extract(
                active,
                text=row.text,
                shots=shot_pairs,
                retries=config.retries,
                decode=Decode(
                    system=system,
                    constrained=config.constrained,
                    schema=task_schema,
                ),
                task=task,
            )
        elapsed += perf_counter() - start
        if isinstance(outcome, VerifyFail):
            reasons.extend(outcome.reasons)
        scores.append(_score_result(gold=row.outcome, result=outcome, task=task))

    n = max(1, len(rows))
    return Trial(
        config=config,
        scores=tuple(scores),
        latency_ms=elapsed / n * 1000.0,
        cost_usd=max(0.0, getattr(meter, "usd", 0.0) - cost_before) / n,
        reasons=tuple(reasons),
    )


def _voted_outcome(  # noqa: PLR0913 - the loop's own knobs
    completer: Completer,
    *,
    text: str,
    shots: tuple[tuple[str, object], ...],
    config: HarnessConfig,
    system: str,
    task: Task = RESTAURANT,
) -> VerifyResult:
    """Sample the same call N times and vote per field.

    Self-consistency: a 7B's mistakes on `intent` and `status` are not stable
    across samples, so the mode is more reliable than any single draw. Voting on
    closed fields only; free text keeps the sample that carried the winning vote.
    """
    passing: list[RestaurantOutcome] = []
    saw_fail = False
    for _ in range(config.consistency):
        result = extract(
            completer,
            text=text,
            shots=shots,
            retries=config.retries,
            decode=Decode(
                system=system,
                constrained=config.constrained,
                temperature=_CONSENSUS_TEMPERATURE,
            ),
            task=task,
        )
        if isinstance(result, VerifyOk):
            passing.append(result.outcome)
        else:
            saw_fail = True
    if not passing:
        reason = "no sample cleared the gate" if saw_fail else "no samples"
        return VerifyFail(reasons=(reason,))
    return _consensus(passing)


def _consensus(outcomes: list[RestaurantOutcome]) -> VerifyResult:
    """Mode per field, then re-run the gate on the merged outcome."""
    if len(outcomes) == 1:
        return parse_and_verify(outcomes[0].model_dump_json())
    merged: dict[str, object] = {}
    for name in _CLOSED_FIELDS:
        merged[name] = _modal([getattr(o, name) for o in outcomes])
    for name in _PHRASE_FIELDS:
        merged[name] = _modal([getattr(o, name) for o in outcomes])
    for candidate in outcomes:
        same = (
            candidate.intent == merged["intent"]
            and candidate.status == merged["status"]
        )
        if same:
            merged["evidence"] = candidate.evidence
            break
    else:
        merged["evidence"] = outcomes[0].evidence
    # A vote can assemble a combination no single sample produced. Re-gate it.
    voted = parse_and_verify(json.dumps(merged))
    if isinstance(voted, VerifyOk):
        return voted
    return parse_and_verify(outcomes[0].model_dump_json())


def _modal(values: list[object]) -> object:
    """Most common value; ties go to the first occurrence."""
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda v: counts[v])


def pareto_front(trials: tuple[Trial, ...]) -> tuple[Trial, ...]:
    """Nondominated trials: nobody is better on quality, cost, and latency at once."""
    front: list[Trial] = []
    for a in trials:
        dominated = any(
            b.quality >= a.quality
            and b.cost_usd <= a.cost_usd
            and b.latency_ms <= a.latency_ms
            and (b.quality, -b.cost_usd, -b.latency_ms)
            != (a.quality, -a.cost_usd, -a.latency_ms)
            for b in trials
        )
        if not dominated:
            front.append(a)
    return tuple(front)


def pick(front: tuple[Trial, ...]) -> Trial:
    """Winner from the Pareto front: quality first, then cheap, then fast.

    Ties prefer the cheaper harness, then fewer shots, so a compile does not
    spend k-shot tokens it does not need.
    """
    return max(
        front,
        key=lambda t: (t.quality, -t.cost_usd, -t.latency_ms, -t.config.k_shot),
    )


def _rung_size(dev: tuple[ExampleRecord, ...]) -> int:
    """How many dev rows the first ASHA rung gets."""
    return min(_MINIBATCH, len(dev))


def _prune_keep(n: int) -> int:
    """How many candidates survive a rung (PLAN §15.7, eta=2)."""
    return max(1, n // _PRUNE_ETA)


def search(  # noqa: PLR0913 - the search entry point names its knobs
    completer: Completer,
    *,
    shots: tuple[ExampleRecord, ...],
    dev: tuple[ExampleRecord, ...],
    space: SearchSpace,
    budget: Budget | None = None,
    task: Task = RESTAURANT,
    bootstrapped: tuple[tuple[str, object], ...] = (),
    models: Mapping[str, Completer] | None = None,
) -> tuple[Trial, tuple[Trial, ...], bool]:
    """Search on dev. Returns (winner, every measured trial, stopped_early).

    Two rungs when the dev slice is big enough to afford one: every candidate is
    priced on a minibatch, then only survivors earn a full dev pass. Both loops
    check the SLO after each candidate, so a compile that has already won stops
    instead of spending the rest of its budget (PLAN §15.12).
    """
    spend = budget or Budget()
    chosen = space.sample(spend.trials, seed=spend.seed)
    seen: list[Trial] = []
    full: list[Trial] = []
    stopped = False

    def clears(trial: Trial) -> bool:
        return spend.slos is not None and trial.meets(spend.slos)

    staged = spend.staged and len(dev) > _STAGING_FLOOR and len(chosen) > 1

    if staged:
        rung = dev[: _rung_size(dev)]
        priced: list[Trial] = []
        for config in chosen:
            trial = evaluate(
                completer,
                rung,
                config,
                shots=shots,
                task=task,
                bootstrapped=bootstrapped,
                models=models,
            )
            seen.append(trial)
            priced.append(trial)
            if not trial.violations and clears(trial):
                confirmed = evaluate(
                    completer,
                    dev,
                    config,
                    shots=shots,
                    task=task,
                    models=models,
                )
                seen.append(confirmed)
                full.append(confirmed)
                stopped = not confirmed.violations and clears(confirmed)
                break
        if not stopped:
            ranked = sorted(
                priced,
                key=lambda t: (t.violations == 0, t.quality),
                reverse=True,
            )
            for trial in ranked[: _prune_keep(len(ranked))]:
                survivor = evaluate(
                    completer,
                    dev,
                    trial.config,
                    shots=shots,
                    task=task,
                    bootstrapped=bootstrapped,
                    models=models,
                )
                seen.append(survivor)
                full.append(survivor)
                if not survivor.violations and clears(survivor):
                    stopped = True
                    break
    else:
        for config in chosen:
            trial = evaluate(
                completer,
                dev,
                config,
                shots=shots,
                task=task,
                bootstrapped=bootstrapped,
                models=models,
            )
            seen.append(trial)
            full.append(trial)
            if not trial.violations and clears(trial):
                stopped = True
                break

    clean = [t for t in full if not t.violations]
    ranked = clean or full or seen
    winner = pick(pareto_front(tuple(ranked)))
    return winner, tuple(seen), stopped


def _score_result(
    *, gold: object, result: VerifyResult, task: Task = RESTAURANT
) -> Scored:
    match result:
        case VerifyOk(outcome=pred):
            return task.score_pair(gold, pred)
        case VerifyFail():
            return ExampleScore(
                schema_ok=False,
                field_hits=0,
                field_total=len(task.scored_fields),
                line_f1=0.0,
            )
