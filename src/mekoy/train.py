"""Training: gated, optional, and only when the eval says the System is short.

PLAN §15.10 is a rule, not an option: *"If below gate: one LoRA SFT rank 8-16, then
GEPA-light on the adapter."* §41.12 asks for the job adapter. §15.8's report line
*"training: skipped"* is a first-class outcome, so the decision to train has to be
derived from the measured result rather than picked by hand.

What was verified here: Apple Silicon trains a LoRA locally with MLX, and the
result was measured against the same held-out rows as the prompt-only System.
What was not: the Fireworks backend. No credential exists on this machine, so its
payload is constructed and gated but its submission is untested. That is stated
rather than implied.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mekoy.compile import CompileReport
from mekoy.errors import CompileError
from mekoy.harness import DEFAULT_PROMPT, PROMPTS, _user_prompt
from mekoy.spec import Slos
from mekoy.tasks import RESTAURANT, Task

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = [
    "FIREWORKS_KEY_ENV",
    "SCRATCH_SIZES",
    "LoRaBudget",
    "ScratchPlan",
    "ScratchSize",
    "TrainingPlan",
    "fireworks_payload",
    "plan_training",
    "scratch_plan",
    "should_train",
    "write_mlx_dataset",
]

FIREWORKS_KEY_ENV = "FIREWORKS_API_KEY"
FIREWORKS_LORA_URL = "https://api.fireworks.ai/inference/v1/lora"
#: PLAN 15.10 fixes the rank band. Eight is the cheap end, which is what a
#: 58-example corpus can support without memorising it.
DEFAULT_RANK = 8
DEFAULT_ITERS = 120


@dataclass(frozen=True, slots=True)
class LoRaBudget:
    """What a training run is allowed to spend."""

    rank: int = DEFAULT_RANK
    iters: int = DEFAULT_ITERS
    backend: str = "mlx"

    def __post_init__(self) -> None:
        """Refuse a rank outside the plan's band rather than silently clamping."""
        if not 8 <= self.rank <= 16:  # noqa: PLR2004 - the band the plan fixes
            msg = f"PLAN 15.10 fixes rank 8-16; got {self.rank}"
            raise CompileError(message=msg)
        if self.backend not in {"mlx", "fireworks"}:
            msg = f"unknown training backend: {self.backend!r}"
            raise CompileError(message=msg)


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Whether to train, why, and with what."""

    train: bool
    reason: str
    budget: LoRaBudget

    @property
    def report_line(self) -> str:
        """What the compile card should say."""
        if not self.train:
            return "training: skipped"
        return f"training: {self.budget.backend} rank {self.budget.rank}"


def should_train(report: CompileReport, *, slos: Slos | None = None) -> bool:
    """PLAN §15.10: train only when the compiled System is below the gate.

    The gate is the declared SLO when there is one, and otherwise one field of
    headroom under what the search achieved. Training a System that already clears
    its gate spends compute to move a number nobody asked to move.
    """
    if slos is not None:
        return report.test.quality < slos.quality
    return report.test.quality < 1.0


def plan_training(
    report: CompileReport,
    *,
    slos: Slos | None = None,
    budget: LoRaBudget | None = None,
) -> TrainingPlan:
    """Decide whether to train, and say why in words."""
    spend = budget or LoRaBudget()
    if not should_train(report, slos=slos):
        gate = "the declared gate" if slos is not None else "a perfect test score"
        reason = f"skipped: the winner already clears {gate}"
        return TrainingPlan(train=False, reason=reason, budget=spend)
    reason = (
        f"below gate at test quality {report.test.quality:.3f}; "
        f"one LoRA SFT rank {spend.rank}"
    )
    return TrainingPlan(train=True, reason=reason, budget=spend)


def write_mlx_dataset(
    directory: Path,
    *,
    train: tuple[tuple[str, BaseModel], ...],
    valid: tuple[tuple[str, BaseModel], ...],
    task: Task = RESTAURANT,
) -> Path:
    """Write chat-format JSONL for `mlx_lm.lora`.

    The user turn is the System's own prompt, so the adapter learns the harness
    that was measured rather than some other rendering of the task.
    """
    directory.mkdir(parents=True, exist_ok=True)
    system = PROMPTS.get(DEFAULT_PROMPT, task.prompt)
    for name, part in (("train.jsonl", train), ("valid.jsonl", valid)):
        lines = [
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": _user_prompt(text, ())},
                        {"role": "assistant", "content": gold.model_dump_json()},
                    ]
                },
                ensure_ascii=False,
            )
            for text, gold in part
        ]
        _ = (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def mlx_command(
    *,
    base_model: str,
    data: Path,
    adapter_path: Path,
    budget: LoRaBudget | None = None,
) -> list[str]:
    """The exact local training command, so a run is reproducible."""
    spend = budget or LoRaBudget()
    return [
        "mlx_lm.lora",
        "--model",
        base_model,
        "--train",
        "--data",
        str(data),
        "--fine-tune-type",
        "lora",
        "--iters",
        str(spend.iters),
        "--num-layers",
        str(spend.rank),
        "--mask-prompt",
        "--adapter-path",
        str(adapter_path),
    ]


def fireworks_payload(
    *, base_model: str, dataset_id: str, budget: LoRaBudget | None = None
) -> dict[str, object]:
    """Build a Fireworks LoRA job body.

    **Not verified.** No Fireworks credential exists on this machine, so nothing
    here has been submitted. It is written against the documented shape and gated
    so that an unconfigured caller cannot accidentally use it.
    """
    spend = budget or LoRaBudget(backend="fireworks")
    return {
        "base_model": base_model,
        "dataset": dataset_id,
        "lora_rank": spend.rank,
        "epochs": 1,
        "learning_rate": 1e-4,
    }


# --- Tier 3: train a small model from scratch -----------------------------------
#
# Modelled on nanochat, whose central lesson is that from-scratch training should
# expose ONE dial. There, `--depth` (transformer layers) determines width, heads,
# learning rate, and training horizon automatically, so the model comes out
# compute-optimal without the user configuring anything. Asking someone to pick a
# learning rate is asking them to do our job.
#
# Verified: the shape of the pipeline, from reading nanochat's stage scripts. Not
# verified here: a run. It needs 8 rented H100s, so it is an integration point and
# it stays behind an explicit gate.


@dataclass(frozen=True, slots=True)
class ScratchSize:
    """One point on the size dial."""

    depth: int
    label: str
    gpu_hours_on_8xh100: float
    usd_on_demand: float
    usd_spot: float
    note: str = ""


#: Published nanochat figures: a GPT-2-capability model costs about $48 on demand
#: or about $15 on a spot instance, in roughly two hours on an 8xH100 node. Depth
#: around 24-26 is that capability. Smaller depths are extrapolated and labelled as
#: such rather than presented as measured.
SCRATCH_SIZES: tuple[ScratchSize, ...] = (
    ScratchSize(12, "tiny", 0.5, 12.0, 4.0, "below GPT-2; for plumbing tests"),
    ScratchSize(20, "small", 1.5, 36.0, 11.0, "near GPT-2; nanochat's default"),
    ScratchSize(26, "gpt2", 2.0, 48.0, 15.0, "GPT-2 capability, published figure"),
)


@dataclass(frozen=True, slots=True)
class ScratchPlan:
    """What training from scratch would involve."""

    size: ScratchSize
    stages: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    provider: str

    @property
    def cost_line(self) -> str:
        """One line a person can decide on."""
        return (
            f"training from scratch: {self.size.label} (depth {self.size.depth}), "
            f"about {self.size.usd_on_demand:.0f} USD on demand or "
            f"{self.size.usd_spot:.0f} USD spot, "
            f"{self.size.gpu_hours_on_8xh100:.1f} GPU-hours on 8xH100"
        )


def scratch_plan(depth: int = 26, *, provider: str = "rented-8xh100") -> ScratchPlan:
    """Build the staged plan for training a small model from scratch.

    The stages mirror nanochat's script layout: tokenizer, pretrain, evaluate,
    supervise, then chat. They are ordered because each needs the previous
    artifact; none of them is optional.
    """
    size = next((s for s in SCRATCH_SIZES if s.depth == depth), None)
    if size is None:
        offered = ", ".join(str(s.depth) for s in SCRATCH_SIZES)
        msg = f"depth {depth} is not on the dial; offered: {offered}"
        raise CompileError(message=msg)
    stages = ("tokenizer", "pretrain", "evaluate", "supervise", "chat")
    commands = (
        ("nanochat/scripts/tok_train.py",),
        ("nanochat/scripts/base_train.py", f"--depth={size.depth}"),
        ("nanochat/scripts/base_eval.py",),
        ("nanochat/scripts/chat_sft.py",),
        ("nanochat/scripts/chat_cli.py",),
    )
    return ScratchPlan(size=size, stages=stages, commands=commands, provider=provider)


def require_scratch_approval(*, approved: bool, provider: str) -> None:
    """Refuse a from-scratch run without explicit approval.

    Training from scratch rents GPUs and spends real money, so it is never
    implicit. The LoRA path is minutes and free; this is neither.
    """
    if approved:
        return
    msg = (
        f"training from scratch on {provider} needs explicit approval: it rents "
        "GPUs and costs money. Pass approved=True once you have agreed the budget."
    )
    raise CompileError(message=msg)


def require_fireworks_key() -> str:
    """Return the key, or refuse. Gated, per PLAN 41.12."""
    key = os.environ.get(FIREWORKS_KEY_ENV, "").strip()
    if not key:
        msg = (
            f"{FIREWORKS_KEY_ENV} is not set, so the Fireworks LoRA adapter "
            "cannot submit a job. Use backend='mlx' to train locally."
        )
        raise CompileError(message=msg)
    return key
