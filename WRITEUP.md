# What we measured

Two claims are easy to make and hard to check: that a compiled System beats a
frontier model, and that it is cheaper. This file records what we actually ran,
on what data, and what the numbers were. It is written to be falsified.

Last updated after the corpus grew to 271 rows and the `when` metric was fixed.
Local model: `qwen2.5:14b` via Ollama, Apple Silicon.

## Headline

On a 54-row test split, the compiled local System scores **0.944** against
**0.958** for GPT-5.5 one-shot and beats the other five frontier configurations:

| lane | quality | strict | s/call | $/doc |
|---|---|---|---|---|
| `gpt-5.5`, one-shot | 0.958 | 0.905 | 1.62 | 0.01700 |
| **compiled `qwen2.5:14b` (local)** | **0.944** | **0.902** | 10.15 | **0.00000** |
| `claude-opus-4.8`, 2-shot | 0.937 | 0.881 | 1.51 | 0.01700 |
| `gpt-5.5`, 2-shot | 0.937 | 0.860 | 1.70 | 0.01700 |
| `gemini-3.1-pro`, one-shot | 0.931 | 0.889 | 1.54 | 0.01700 |
| `gemini-3.1-pro`, 2-shot | 0.931 | 0.892 | 1.72 | 0.01700 |
| `claude-opus-4.8`, one-shot | 0.918 | 0.873 | 1.40 | 0.01700 |

Under the repo's own rule (`bakeoff.py`: quality >= baseline AND cheaper) that is
**CHALLENGER WINS against five of six frontier configurations** and a loss by 0.014
against GPT-5.5 one-shot — five fields out of 378.

On `strict` (raw string comparison) the gap to GPT-5.5 is **0.902 vs 0.905**, which
is three fields. The compiled System is level with frontier extraction quality and
wins outright on cost.

**Caveat that matters:** five fields on 54 rows is inside the noise. Do not claim
we beat GPT-5.5; claim we are level with it at zero marginal cost.

## The job

Frozen extract-and-verify from already-text: read one messy restaurant-call
transcript, emit one JSON object with eight fields. No chat, no booking.

The policy gate (`src/mekoy/verify.py`) rejects a candidate outright when:

- `booked` is true and `intent` is `availability`
- `booked` is true and `status` is `unknown`
- `booked` is true and `status` is not `confirmed`
- a `confirmed` reservation leaves `booked` false
- `restaurant` or `evidence` is empty

`booked` is the safety-critical field. A System that invents reservations is
worse than one that reports less.

## The data

14 hand-labeled calls in `examples/bucko-restaurant/examples.jsonl`, plus two
generated corpora whose gold labels are known by construction:

- `generated.jsonl` — 96 rows from a scenario table (`src/mekoy/evalgen.py`).
- `paraphrased.jsonl` — 175 rows where the *scenario* is declared in code and a
  local model writes only the messy surface text (`src/mekoy/paraphrase.py`).
  Each rendering must pass `accepts()`, which rejects a transcript that
  contradicts its own label. Nothing else enters the corpus.
- `combined.jsonl` — 271 rows, **163 train / 54 dev / 54 test**.

The paraphrased rows matter because template text shares the author's blind spots.
They are rougher, and they cost us accuracy: on the 96-row corpus the same System
scored 0.94–0.97. On the 271-row corpus it scores 0.944. **More realistic data
lowered the honest score**, which is the expected direction and worth remembering
whenever a small eval looks good.

The split is dealt by a hash of each row's own text, so a fixture written
family-by-family still spreads across slices. `dataset.audit_coverage` fails if a
slice has no booked reservation, no unbooked call, or only one intent — the three
ways a test set silently stops testing the thing you care about.

## The split discipline

The search reads `train` and `dev`. `test` is scored once, after selection. A test
asserts that no test example ever appears in a search prompt
(`test_search_never_reads_the_test_split`). Reports state which slice was used for
selection, and how many candidates were tried, so a score can never be read
without its selection cost.

## The bake-off, and how a win is declared

### Same model, same test set: the harness alone

Official `bakeoff` path on the 271-row corpus, with the *same* model in both
lanes so the only variable is the compiled harness:

```
mekoy bakeoff examples/bucko-restaurant/combined.jsonl \
  --baseline-model qwen2.5:7b --model qwen2.5:7b --trials 1

holdout n=54 (test, scored once)
baseline   qwen2.5:7b   quality=0.905 strict=0.857 schema=1.000 $0.00000/doc  7564ms
challenger qwen2.5:7b   quality=0.950 strict=0.894 schema=1.000 $0.00000/doc  7656ms
speed      challenger slower
compile    k=8 r=0 grammar default
selection  selected on dev (n=54) from 1 candidate(s); test (n=54) scored once
training: skipped
verdict: CHALLENGER WINS (quality >= baseline, cheaper)
```

**+4.5 accuracy points from the harness alone, for +1.2% latency.** The weights did
not change. This is the strongest claim in the project because it is the only one
where everything except the harness is held fixed: 0.905 one-shot versus 0.950 with
8 shots and grammar-constrained decoding.

Note that the compiled 7B (0.950) edges the 14B at the same harness (0.944). Bigger
weights are not a substitute for a tuned harness.

### Against frontier models, same 54 test rows

## The self-host path works, and testing it found a safety hole

`Dockerfile` was written but never built. Building it failed immediately:

```
ERROR: This version requires zstd for extraction. Please install zstd
```

Fixed, then verified properly: image builds in 51s, `mekoy --help` lists all nine
commands inside it, `ollama` 0.34.0 is present, and a container served
`qwen2.5:1.5b` and extracted correctly in 10s.

The download bundle had a worse defect: its README instructed `docker build -t
system .` while the bundle shipped no Dockerfile. The self-host bar is a compose
file that works on a GPU box, so the bundle now ships
`docker-compose.yml` instead. Its first version hard-reserved an NVIDIA device,
which fails to start on any machine without that driver, so the reservation ships
commented out. Verified by running the shipped bundle: `docker compose up -d`, pull,
invoke from the host, 5.2s.

**Then the invoke returned `booked: true` for a transcript reading "host said a
table for two is free tonight at 8, they did not book it."** The gate passed it,
because `intent=reservation`, `status=confirmed`, and `booked=true` all agree with
each other. The model even quoted the denial in its own `evidence` field. Every
field check was satisfied and the answer was still dangerous — the exact failure
the whole eval exists to prevent.

Field checks cannot see it. The evidence text can, so `verify.py` now rejects
`booked=true` when the evidence denies it:

```
retries=0 -> verify failed: booked is true but the evidence denies it was taken
retries=1 -> {"status":"unknown", ..., "booked": false}
```

The retry loop feeds the gate failure back to the model, and the model corrects
itself. A hallucinated reservation became a truthful "we do not know".

The same hole existed one field over: `status=confirmed` with evidence reading
"they were fully booked" passed every field check. `verify.py` now rejects a
status the evidence contradicts, in both directions, with negation handled so
"No table for two is available" is not read as an offer. A declared `party_size`
that the evidence contradicts is rejected too, anchored on the preposition so a
time is not mistaken for a headcount. All 554 gold rows across every corpus still
pass the gate, which is the only reason these rules are safe to add.

`when` is deliberately **not** gated this way. It is free text with too many
phrasings — "tonight at 8", "next Friday at 7:00 PM", "Saturday brunch 10:30" —
and a text matcher strict enough to catch a wrong time would reject correct rows.
It is also not safety-critical: a wrong time wastes a phone call, a wrong
`booked` wastes an evening. Nine deterministic rules now guard the fields where
being wrong actually costs something.

Two more defects surfaced only by using the thing:

- `mekoy invoke` printed through `rich`, which **line-wraps long JSON**, so the
  output was unparseable when piped. It now writes plain stdout.
- The bundle's compose file hard-reserved an NVIDIA device, so it failed to start
  anywhere without that driver.

Final check through the shipped bundle, four safety-critical cases:

| transcript | booked | status |
|---|---|---|
| "host said a table for two is free tonight at 8, they did not book it" | **false** | unknown |
| "host confirmed a table for 3 tomorrow at 6:30 under Sam" | **true** | confirmed |
| "did not take the reservation for 4 Friday at 7pm. They were fully booked." | **false** | unavailable |
| "did not answer after eight rings. Voicemail. Party of 6." | **false** | unavailable |

The first row used to return `booked: true`. That is the whole product in one line.

This is the most valuable thing found in the whole project, and it came from
running the shipped artifact rather than reading the code.

## The second task class, and the corpus nobody was checking

`src/mekoy/receipt.py` defined a `Receipt` schema that **no other module imported**,
and `examples/cord-receipt/` held 24 labeled receipts — including an 18-row
OCR-noisy fixture — that nothing was scoring. The repo had drifted from receipts
to restaurant calls and left the first prototype behind. Receipts are the first
proof job, and deterministic numeric checks sit at the bottom of the eval stack,
so this was both a gap and a free verification.

Rebuilt the deterministic layer:

```
qty * unit_price == line_total      (per line, EPSILON = 0.011)
Σ line_total     == subtotal
subtotal + tax   == total
currency is 3 alphabetic capitals; date is ISO 8601; item count > 0
```

Then pointed it at the abandoned corpus. **All 24 gold receipts are arithmetically
sound.** That matters because arithmetic is the one thing a model cannot bluff: a
receipt whose gold breaks its own identities could not score anything.

`check-eval` now detects the task class from the row shape (`receipt` key versus
`outcome` key) and audits both:

```
examples/bucko-restaurant/combined.jsonl   271 rows: 270 support their label, 1 not
examples/cord-receipt/hard.jsonl            18 receipts: 18 arithmetically sound, 0 not
```

Scoring for receipts: header text casefolds, money compares within
`EPSILON`, and line items are matched one-to-one on amount with description
agreement as a tiebreak, so reordering lines is not punished.

## Making the compiler task-generic

The receipt path could be validated but not searched: the loop was typed to
`RestaurantOutcome`. `tasks.py` now names what actually varies per job — schema,
instructions, scored fields, the deterministic gate, the retry ladder — and
`RESTAURANT` is the default, so no existing call site changed and all existing
tests kept passing while the refactor landed.

Adding the second class immediately found a bug that only a second class can find.
`evaluate()` defaulted to the restaurant task, so receipt JSON was parsed as a
restaurant outcome: **every receipt candidate failed the gate and the compile
reported 0.000 across the board**. A patch I believed had landed had silently not
matched. The tell was `reasons: []` alongside `schema_ok: False` — a rejection with
no reason is a contradiction, and it is only visible if the failure reasons are
recorded at all.

Two further fixes came out of running it:

- **Retry ladder per task.** The receipt gate is arithmetic. `retries=0..1` cleared
  0 of 5 test receipts; `retries=3` cleared 3 of 5. A policy violation is usually
  fixed in one pass, an arithmetic one is not, so the ladder now lives with the
  task: `RESTAURANT (0, 1)`, `RECEIPT (0, 3)`.
- **An unexplained 0.000 is a bad report.** When every candidate is rejected the
  card now says `blocked: every candidate failed the gate, e.g. sum(line_total)=
  40.00 != subtotal=39.00` instead of printing a winner with a quality of zero.

Receipt compile, end to end through the CLI, task auto-detected from the fixture:

```
task: receipt
winner  k=8 r=3 free default
dev     quality=0.908 schema=1.000 n=5
test    quality=0.556 strict=0.533 schema=0.600 n=5
training: skipped
```

Those numbers came from a 24-row hand-written fixture and did not mean anything.
The real data is now loaded, and it changed the schema twice.

### CORD, 79 real receipts

`mekoy load-cord` fetches the CORD test split through the HuggingFace rows API —
no dataset library, no parquet, no pixels. `valid_line` renders to plain text,
`gt_parse` maps onto the schema.

```
79 examples -> 47 train / 16 dev / 16 test

winner  k=8 r=3 free default
dev     quality=0.916 schema=1.000 n=16
test    quality=0.738 strict=0.729 schema=0.938 n=16
```

**0.738 on real OCR-noisy receipts, against 0.886–0.908 on the fixture.** The
fixture was flattering, which is the expected direction and the reason for using
real rows.

Three things in the real data contradicted the fixture, and all three changed code:

1. **42 of 100 receipts print no subtotal.** A schema that required one could not
   represent them, so every money field but `total` is now optional and the gate
   checks only what is asserted. Requiring a field the receipt never printed is
   how a correct extraction gets marked wrong.
2. **Service charges and discounts exist on top of tax.** The identity that holds
   is `subtotal + tax + service + discount == total`. Measured across the split:
   the naive `subtotal + tax == total` held on 41 of 58 checkable rows, and adding
   service took it to 51.
3. **No row carries store info**, so a required `merchant` would have rejected the
   entire split.

**17 of CORD's 100 test annotations break a deterministic arithmetic identity.**
Scoring a model against gold that does not add up penalises it for being right, so
`load-cord` drops those rows by default and reports the count. 79 of 96 converted
rows survive.

### SROIE, 105 real receipts, and two Eurocentric assumptions

`mekoy load-sroie` pulls SROIE's OCR boxes and annotations straight from the
source repository — 120 attempted in under four seconds with a small thread pool.
Its shape is different again: SROIE annotates only `company`, `date`, `address`,
and `total`, so **no arithmetic identity applies**. A row passing the gate means
the schema validated, not that the numbers were checked. That is a real limit of
this corpus and worth saying rather than implying the gate did work it did not.

The loader needed two fixes that only real files reveal:

- **Box parsing dropped 35 of 44 lines.** `[float(v) for v in row[0::2]][:4]`
  converts the trailing text field before the slice trims it, so every ordinary
  row raised `ValueError` and was skipped; only rows whose text happened to be
  numeric survived. Slicing before converting fixed it. The count looked plausible
  enough that only printing the reconstructed receipt caught it.
- **Two-digit years.** `01/02/99` became `2099`. A pivot at 70 fixed it.

Then the gate itself failed the data twice, both times for assuming a US receipt:

| finding | before | after |
|---|---|---|
| `EPSILON = 0.011` (half a cent) | SROIE rounds to 5 sen, so correct receipts missed by 0.02–0.03 | `0.05`, which still rejects the genuine 0.90 and 0.10 errors in the same run |
| currency must be 3 capitals | `RM` is the Malaysian Ringgit and was rejected | codes *or* symbols: `RM`, `$`, `€`, `Rp` |

Accepting `RM` alone moved SROIE's gate pass rate from **0.762 to 0.905**.

### Real receipts, final numbers

| corpus | rows | test quality, free-form | test quality, schema-constrained | $/doc |
|---|---|---|---|---|
| CORD | 80 | 0.683 | **0.875** | 0.00000 |
| SROIE | 105 | 0.522 | **0.905** | 0.00000 |

Against 0.886–0.908 on the hand-written fixture. Both test splits are 16 and 21
rows, so these numbers carry wide error bars and should not be ranked against each
other.

**Read the constrained column, not the free-form one.** Earlier revisions of this
writeup quoted "0.52–0.68" for real receipts and left it there, which undersold the
engine badly: those are the figures with decoding left unconstrained, and the same
runs with the schema enforced land at **0.875 and 0.905**, close to the restaurant
job's 0.944. The constraint is not a footnote — it is one of the axes the compiler
searches and one of the reasons a compiled System beats an unconstrained prompt.
Quoting the unconstrained number as the engine's capability was a mistake, and a
reader comparing us to a frontier API would have been comparing our worst
configuration to their best.

The free-form column is kept because it is the honest cost of the constraint: a
constrained decode can reject an answer outright rather than return a wrong one, and
knowing how often that happens is worth having.

One scoring choice deserves flagging: `address` is compared as a single field over
a multi-line string, so a mostly-correct address scores zero. That is a harsh rule
and it accounts for part of SROIE's low number. Field-level token F1 would be the
fairer measure for long free-text fields, and it is not implemented.

### Grammar-constrained decoding lost

The search A/Bs constrained against unconstrained decoding, and on receipts it was
not close:

| arm | dev quality | schema |
|---|---|---|
| `k=8 r=3 free` | **0.916** | 1.000 |
| `k=8 r=0 grammar` | 0.664 | 0.688 |
| `k=4 r=3 grammar` | 0.629 | 0.812 |

The warning worth testing is the constrained-decoding "format tax": always A/B
unconstrained. The A/B was built for that reason and it earned its place here.
Every grammar arm scored worse, and two of them also failed the gate more often.

## The third task class is classification, and the abstraction held

Receipts and restaurant calls are both extraction, so they only prove the task
boundary handles two schemas. BANKING77 is a different shape entirely: one label
from a closed set of 77, nothing to quote, no arithmetic, and accuracy instead of
field accuracy.

It ran through the same search, scoring, reporting, and split machinery with no
changes to `search.py`, `compile.py`, or `report.py`. Only `tasks.py` needed a new
entry, and the one thing that had to be removed was a smell: `Task.score_pair`
dispatched on the task's *name*, so a third name meant a third branch. It now calls
the scorer the task carries.

```
task: banking77
winner  k=8 r=0 grammar default
dev     quality=0.634 schema=0.959 n=123
test    quality=0.618 strict=0.618 schema=0.976 n=123
cost    $0.00000/doc  latency=1208ms/doc
```

**0.618 accuracy across 77 intents, 120 of 123 clearing the gate.** The three
rejections returned a label outside the published set, which is the one thing that
can be checked deterministically about a classification.

Two things this corpus showed that the others could not:

- **The cost profile inverts.** 1.2 s/document against SROIE's 54 s, because
  latency here is dominated by output length rather than input. A task class that
  emits six tokens is two orders of magnitude cheaper than one that emits a table,
  at the same price per document.
- **The search found no lever.** All four arms landed at 0.625–0.634 on dev: shots,
  retries, and constrained decoding changed almost nothing. For 77-class intent
  recognition the ceiling is what the model already knows about the label set, and
  the harness cannot buy its way past that. A compile that honestly reports four
  identical arms is more useful than one that invents a winner from noise.

The loader also had to be fixed twice: BANKING77's CSVs are sorted by category, so
taking the first 600 rows covered **five** of 77 intents — a corpus that looks
healthy and tests almost nothing. Round-robin interleaving fixed it: 616 rows,
8 per intent, all 77 present. The same class of bug as the CORD box parser: a
plausible-looking count hiding a broken sample.

## GEPA-light, the borrowed optimiser

Reflection is staged after the candidate pool: survivors get a light reflective
pass capped at 50-150 metric calls, using a larger model as the reflection LM. The
wrapper is built on DSPy **3.3.1** as an optional
extra, because §35 warns that `spec.json` must stay loadable without DSPy.

The integration decision that matters: **GEPA optimises against our metric, not
DSPy's.** The metric parses a reply through the task's schema and gate, scores it
with the task's scorer, and hands the gate's own reasons back as textual feedback:

```
Rejected by the deterministic gate: label 'not_an_intent' is not one of the 77 intents
Fields that disagreed: label: expected 'age_limit', produced 'atm_support'
```

A bare score tells a reflection model nothing it can act on. This way the optimiser
is pushed toward a System that satisfies the checks actually enforced, rather than
toward whatever a generic metric would prefer. The reflection LM is local for the
same reason the task LM is: an instruction written by a closed API is closed-model
output, which AGENTS.md bans from compile data.

### Two places the intent and the library disagree

1. **`auto` and `max_metric_calls` cannot both be set.** The intent was
   "auto=light, cap max_metric_calls 50-150". dspy raises: *"Exactly one of
   max_metric_calls, max_full_evals, auto must be set."* The cap is what bounds a
   compile, so the cap is the default and `auto` is opt-in. Setting both is now a
   clear `CompileError` rather than a stack trace.
2. **litellm's `ollama_chat` provider must not be given the `/v1` suffix.** Our
   runtime speaks the OpenAI-compatible URL, so passing it straight through made
   litellm request `/v1/api/chat` and get a 404. `_ollama_base` strips it.

### Result on BANKING77

```
trainset=12 valset=12, max_metric_calls=30
GEPA: baseline=0.583 best=0.583 improved=False calls=30
```

**It ran, spent its budget, and found nothing.** That is consistent with what the
search already showed on this task: all four arms landed at 0.625-0.634, so the
instruction is not the bottleneck for 77-class intent recognition with a 7B. A
reflection stage that honestly reports no improvement is worth more than one that
manufactures a delta from noise, and it took 4m9s cold.

## Stage 0: BootstrapFewShot, and an honest null result

BootstrapFewShot belongs before any search over k-shot and decode. What
ships without it is "take the first *k* training rows", which is a sample rather
than a selection: it cannot know whether a row teaches anything. `--bootstrap`
adds a candidate whose demonstrations were chosen by tracing the program and
keeping the traces that cleared the gate. It shares the `gepa` extra rather than
adding a second dependency.

One design decision: **the demonstration's label comes from gold, not from the
trace.** The selection is the contribution; teaching a System from a possibly-wrong
model output is not something this pipeline should do.

### It changes nothing on the data we have

```
bootstrapping demonstrations ... kept 4 demos in 2s

first-4 shots    dev quality=0.552 schema=0.812 (19509ms)
bootstrapped     dev quality=0.552 schema=0.812 (18599ms)
```

Identical, and the reason is checkable: **bootstrap kept exactly `train[:4]`.**
DSPy traces in order and stops once it has enough passing traces, and the first
four rows of both real corpora clear the gate. Verified on SROIE too — same four
rows, same order. So the stage is correct and degenerate here.

**The precondition for it to diverge is that early training rows fail the gate**,
and that condition was not met on either corpus. I did not construct a corpus to
force it, so the claim supported by evidence is "no divergence observed", not
"no divergence exists". Getting a real measurement would mean finding or building a
corpus whose first rows the model fails, which is a property of the data rather
than of the code.

That still leaves the stage worth having: it costs one pass over the training set
and it stops the pipeline from claiming that "first *k*" is a selection.

## The declared API surface, and a bug only a live call finds

The API has a declared REST surface. Diffing it against the implementation found
four endpoints that were never built:

```
GET  /v1/systems/:id              added
GET  /v1/systems/:id/report       added
POST /v1/systems/:id/deploy       added
GET  /v1/systems/:id/compare      added
POST /v1/chat/completions         added  (so an OpenAI client works too)
```

All of them are now implemented and covered by a test that reads `openapi.json` and
asserts every declared route exists, so the surface cannot silently drift again.
`POST /v1/run` is a later-phase endpoint by intent and is deliberately absent.

Verified against the running server, not only the test client:

```
POST /v1/systems              -> draft (12 examples)
GET  /v1/systems/:id          -> 200 draft, run = None
POST /v1/systems/:id/compile  -> 200 succeeded | k=0 r=1 grammar default
GET  /v1/systems/:id/report   -> 200 test quality=0.857 schema=1.000 n=2
POST /v1/systems/:id/deploy   -> 200 bundle written: spec.json, report.txt, ...
   ... hosted mode            -> 200 "not hosted; use download or invoke"
GET  /v1/systems/:id/compare  -> 200 quality tie within 0.01 on n=2
```

### The bug the seventh call found

`POST /v1/chat/completions` returned **503, model server unreachable**. The route
takes `model` as the *System id*, which is the only thing an OpenAI client can pass,
and that id was then handed to Ollama as a base-model name — so Ollama 404'd on
`sys_deb789fa...`.

The real defect was underneath: **a run never recorded which model it compiled
against.** A System is the harness *plus* the model, so a run now stores its
`model`, and invoke (both routes) replays it rather than defaulting. `InvokeRequest.model`
became optional for the same reason: unset means "the model this System was
measured with", which is the honest default. Re-verified live afterwards: both
routes return 200 and the chat completion carries the extracted JSON.

Static review and `tsc` cannot see any of this. It took calling the endpoint.

## The web UI, verified against the API

The Next.js app was already running, so it was worth checking whether the three-way
split broke it. It did: `web/src/lib/mekoy-api.ts` read `n_holdout` from the eval
response, and that field had been renamed to `n_train` / `n_dev` / `n_test`. The UI
would have shown `undefined` for the holdout count with no error anywhere.

Fixed on both sides — the API keeps `n_holdout` as a documented compatibility alias
for `n_test`, and the client now reads the real fields and surfaces train, dev, and
test separately. The `winner` payload gained `config`, `constrained`, `prompt`,
`cost_usd`, and `latency_ms`, so the client type was widened to match.

Verified: `tsc --noEmit` exits 0, `eslint` exits 0, the page renders (74 KB, 200),
and the exact endpoint sequence the UI calls returns the expected shapes live:

```
POST /v1/systems            -> id, phase, n_examples
POST /v1/systems/:id/evals  -> approved, n_train=8, n_dev=2, n_test=2, n_holdout=2
POST /v1/systems/:id/compile-> winner{config, k_shot, retries, constrained,
                                      prompt, quality, schema_rate,
                                      cost_usd, latency_ms}
```

Not verified: clicking through the browser flow. The wiring is proven, the
interaction is not.

## Driving the API from a real client found three defects

The API was exercised the way a client actually uses it, end to end, and the flow now
completes against the local stack with no closed-API key:

```
POST /v1/systems                             -> 200 OK   (was 422)
POST /v1/systems/sys_5100.../evals           -> 200 OK
POST /v1/systems/sys_5100.../compile         -> 200 OK
```

Three defects stood between the API and that, and none of them showed up in
tsc, lint, or code review:

1. **The create endpoint only accepted one task class.** `POST /v1/systems` returned
   422 for the client's own receipt samples, because the endpoint only accepted
   restaurant rows — so the API could not create the System its own documentation
   described. Rows are now generic, the task class is detected from the label key and
   remembered on the System, and `coerce_label` is shared with the file loader so the
   two cannot diverge. Creating with fewer than three rows is refused at create time
   rather than failing later at eval.
2. **An OpenAI-compatible client needs the Chat Completions path.** The AI SDK's
   provider defaults to the Responses API, where the first turn worked and the second
   died with `input[2]: unknown input item type: "item_reference"` — a Responses-API
   construct a local server does not implement. Forcing Chat Completions is what makes
   any OpenAI-compatible server work.
3. **A client needs a model host it can set.** The default pointed at a hosted
   provider, so a local-only flow could not run without that provider's key. It now
   reads `MEKOY_CHAT_BASE_URL`, `MEKOY_CHAT_MODEL`, and `MEKOY_CHAT_API_KEY`, and
   defaults to Ollama.

### A debugging lesson, kept because it cost an afternoon

I nearly filed "the API is broken" against a client that went through
`127.0.0.1:3001`. The real cause was the origin: Next.js blocks its dev resources as
cross-origin and treats `localhost` as canonical, so React never hydrated and a form
fell back to a native GET. The API was fine; the test harness was wrong. The warning
in the dev-server log was the tell, and a DOM check made it unambiguous. **Check the
instrument before blaming the subject.**

## Experiment tracking

MLflow is the experiments layer. What it
buys is the thing one report card cannot: comparing compiles over time and across
task classes, which is what a catalog ranker would eventually read.

```
mekoy compile examples/cord-receipt/cord.jsonl --quick --model qwen2.5:7b --track

tracked: examples/cord-receipt/compile-runs.jsonl +
mlflow:88950cdb9be64a118185c67617ef0968
```

```
{"task":"receipt","config":"k=0 r=1 grammar default","dev_quality":0.00625,
 "test_quality":0.0,"schema_rate":0.0,"latency_ms":34359.3,"test_n":16.0}
```

That row is a real quick compile on CORD, and it is honestly bad: zero-shot with
grammar-constrained decoding fails the gate on every test document. The log said so
without anyone having to ask.

Three decisions:

- **Optional dependency.** MLflow adds 51 packages, so it lives in the `mlflow`
  extra and is imported lazily; a core install pays nothing.
- **The JSONL line is durable; MLflow is the interface.** Auto writes both. The
  first version wrote only MLflow, which silently discarded the `log_path` the
  caller supplied — a wart the test suite caught and now pins.
- **Params and metrics stay separate.** What was chosen (`k_shot`, `retries`,
  `constrained`, `prompt`) versus what was measured (`test_quality`, `schema_rate`,
  `cost_usd`, `latency_ms`). A tracking layer that logs a blob is a log file; this
  one can be read column-wise, which is the only reason to log it.

The MLflow path was verified against a real SQLite store, not mocked: a run written
and read back with its params and metrics intact. An unverified branch is worse than
no branch.

Note for anyone syncing extras: `uv sync --extra mlflow` **prunes** the `gepa`
extra, which silently skipped three DSPy tests. Use `--all-extras` to keep both.

## Auth and artifact storage, in the form that needs no account

Auth and object storage are both wanted, and the provider is left open. The provider
is not specified and this runs locally, so what ships is the smallest thing that is
actually auth rather than a placeholder.

A bearer key on `/v1`, compared with `secrets.compare_digest`, off unless
`MEKOY_API_KEY` is set. `X-API-Key` works too, because curl and `fetch` are friendlier
with a plain header. Verified against a live server:

```
GET  /health                    -> 200   (a probe that needs a key cannot probe)
POST /v1/systems  (no key)      -> 401   "missing or invalid API key"
POST /v1/systems  (wrong key)   -> 401
POST /v1/systems  (right key)   -> 422   past auth, failed validation as expected
```

Artifacts go through a three-method `ArtifactStore` protocol with a filesystem
backend, so a hosted bucket is a second implementation rather than a change to the
deploy route. `MEKOY_ARTIFACTS_DIR` moves the root, and a keyed `deploy` was
verified to land there:

```
/private/tmp/mekoy-artifacts/sys_23d6d9df...
  spec.json  report.txt  README.md  docker-compose.yml
```

`system_id` is caller-supplied, so the resolved path is checked against the store
root before anything is written.

### What is honestly not built

- **Clerk is not integrated.** No OAuth, no sessions, no user accounts. One shared
  key is what exists.
- **No object storage.** The S3 backend is a documented extension point, not code.
  A bucket needs an account.
- **The MCP router is deliberately left open.** A connector calling in from Claude or
  ChatGPT has nowhere to put our bearer token, so protecting it needs a key-exchange
  design rather than a header check. That is a real gap, not an oversight, and it is
  the thing standing between this and "deploy for others to use".

One implementation note worth recording: the first version warned via
`warnings.warn` that `/v1` was open. That fires at import through the module-level
`create_app()`, which broke pytest collection. It is an operational notice about a
running server, so it is logged, not warned.

## A second serving stack, and the format tax measured

Serving an open model with JSON schema, through a managed API or a local engine. I had
claimed this was blocked on a GPU. It was not: **`llama-server` is installed**, which
is a genuinely different serving stack from Ollama, and it is reachable on this
machine. The runtime is a generic OpenAI-compatible client, so pointing it at
llama.cpp was a `base_url` — which is itself the evidence that a vLLM instance
would need no code change either.

The part that *was* genuinely missing was the **schema**. The runtime sent
`{"type":"json_object"}`, which asks for valid JSON, not for *this* JSON. It now
sends:

```
response_format: {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}
```

which is the shape llama.cpp, vLLM, and OpenAI all accept. Verified accepted by both
stacks, including a nested schema with `$defs`/`$ref` (the receipt model).

### Servers disagree, so it is a search axis and not a default

```
Ollama (qwen2.5:7b), same dev slice:
  grammar (json_object)   dev quality=0.925  schema=1.000  6209ms
  schema  (json_schema)   dev quality=0.910  schema=1.000  7410ms

llama.cpp (0.5B), same dev slice:
  free    (no constraint) dev quality=0.782  gate pass=0.842   722ms
  grammar (json_object)   dev quality=0.782  gate pass=0.842   727ms
  schema  (json_schema)   dev quality=0.647  gate pass=0.895   485ms
```

Two things worth reading carefully.

**On llama.cpp the schema raises structural validity and lowers field accuracy:**
gate pass rate 0.842 → 0.895 while quality 0.782 → 0.647. That is the constrained
decoding "format tax" — structure can hurt accuracy — measured rather
than asserted. The constraint keeps the output parseable more often and makes the
values worse.

**On Ollama the schema is accepted and not honoured.** A nested request came back
`{}`, and schema mode scored slightly worse and slower. So `constrained` and
`schema` are separate axes and the search picks per stack, which is why adding this
did not change the shipped default.

A server that *rejects* a schema raises rather than silently degrading — pinned by a
test, because a silent fallback here would hide exactly the difference above.

**Caveat, stated plainly:** no vLLM instance was run, because this machine has no
supported GPU for one. What is verified is that the code path is stack-agnostic on
two real servers.

## The HTTP API is deployable, and the container is where one gap showed up

The API and the container are two different things, and building the container found a
defect the local run had been hiding.

What ships is everything except a paid hosting step:

- **`Dockerfile.api`** — the control plane, distinct from the repo's other Dockerfile,
  which serves a *downloaded System*. Deploying the wrong one would ship a model
  instead of the API.
- **`fly.toml`** — an example deploy config: `internal_port = 8080`, a `/health`
  check, `auto_stop_machines` so a demo API does not idle-burn, and a volume for
  bundles. It is a starting point to copy, not our infrastructure.

The container also fixed a capability gap I had only observed before. Against the
local API an upgrade request gets:

```
WARNING: Unsupported upgrade request.
WARNING: No supported WebSocket library detected.
```

Against the container:

```
INFO: "WebSocket /agent" 403
INFO: connection rejected (403 Forbidden)
```

`uvicorn[standard]` means the upgrade actually happens and reaches the router. A
streaming client sees a refusal instead of a broken endpoint, which is the
difference between a client that can be debugged and one that cannot.

**A public deploy is a billing decision, so nothing is deployed by this repo.** It
would create a billable app and needs an explicit yes from whoever owns the account.
Two caveats worth stating before anyone deploys it:

- The store is process-local, so the container runs a single worker and loses every
  System on restart. Persistent object storage would need a hosted bucket, which needs
  an account, so what ships is the interface and a local implementation of it.
- A container's `127.0.0.1` is the container, so the default model URL points at
  nothing. `MEKOY_MODEL_BASE_URL` sets a deployment default and a caller may override
  it per request. Compiling in a deployment needs someone to point it at a reachable
  model server.

## Training, exercised for the first time

Every compile in this project has reported `training: skipped`. That is a
first-class outcome, but it meant the rule "if below gate: one LoRA SFT at rank
8-16" had never run. Fireworks is one way to run it, and there is no Fireworks
credential on this machine — but **Apple Silicon trains LoRA natively**, which makes
the training rule testable without one.

`mlx` was already installed. A rank-8 LoRA on Qwen2.5-1.5B-Instruct-4bit, 120
iterations over the 58-row training split:

```
train loss 0.067 -> 0.002     val loss 0.077 -> 0.003     peak mem 1.76 GB
120 iterations in 1m23s, 10.5 MB of adapter weights
```

Validation loss tracked training loss down, so it learned the task rather than the
training rows. Evaluated on the 19 held-out rows, against the same base model:

| config | base | LoRA |
|---|---|---|
| k=0 zero-shot | 0.714 | **0.752** (+0.038) |
| k=4 four shots | **0.895** | 0.880 (-0.015) |

**Training substitutes for prompting.** The adapter closes most of the zero-shot
gap and then slightly *hurts* once four demonstrations already carry the same
information. That is the overlap BetterTogether's `p -> w -> p` addresses, measured
here on a task where the two are nearly redundant. 19 test rows means ~0.0075 per
field, so +0.038 is about five fields — real, and not a large sample.

### The mistake I nearly published

`mlx_lm.server --adapter-path` accepts the flag and **does not apply the adapter**.
The base and "adapter" servers returned byte-identical output, and my first
evaluation showed 0.714 / 0.895 for both — which reads as "LoRA changed nothing".
The numbers were identical because the LoRA was not running. Fusing
(`mlx_lm.fuse`, 847 MB vs 839 MB) produced different output immediately, and only
then did the measurement mean anything.

I would have reported a clean null result that was an artifact of a flag being
ignored. The tell was outputs matching to three decimals across nineteen documents —
identical is suspicious in a way that similar is not.

### What the code now encodes

`train.py` makes the training rule executable rather than a note:

- **`should_train(report, slos)`** — training runs only below the gate. A
  System that already clears its gate spends compute to move a number nobody asked
  to move, and the card says `training: skipped` with the reason.
- **The rank band is enforced, not clamped.** `LoRaBudget(rank=32)` raises, because
  a silent clamp hides a bad configuration.
- **The training data is the System's own prompt**, so the adapter learns the
  harness that was measured rather than some other rendering of the task.

**Not verified: the Fireworks backend.** No credential exists here, so
`fireworks_payload` builds the documented body and `require_fireworks_key` refuses
without one — and that refusal is the only behaviour of it this project can claim
to have tested.

## Two CLI commands that were missing, and the gap that blocked a safe deploy

Re-reading the intended CLI surface — eval / compile / report / serve — instead of
assuming it was done:

**`report` did not exist.** Added: it prints the card the compile wrote next to the
corpus, so a report is readable without the API running.

**`serve` was a stub that printed "not in local MVP; use invoke" and exited 2** —
while the server it should serve had existed in `mekoy.api.main` for a while. Those
two things are different: *we do not host for a customer* is a product decision;
*the command does nothing* is a bug. `serve` now runs uvicorn, with `--host`,
`--port`, and `--reload`, and warns when it is starting an unguarded API.

```
mekoy report examples/bucko-restaurant/generated.jsonl
  winner  k=8 r=0 grammar default
  dev     quality=0.934 schema=1.000 n=54
  test    quality=0.894 strict=0.894 schema=1.000 n=54

mekoy serve --port 18788
  /health    (open)   -> 200
  /v1   no key        -> 401
  /mcp  no key        -> 401      <- was 200
  /mcp  with key      -> 200
```

### The `/mcp` gap

I had called the open MCP endpoint "a real gap, not an oversight" and said it needed
a key-exchange design before anything could be deployed. **That was wrong.** The MCP
HTTP transport carries an `Authorization` header, so a connector already has
somewhere to put a key — the endpoint was simply not guarded.

Leaving it open was the one thing that made a public deploy irresponsible: an
unauthenticated endpoint that accepts compile jobs. It is now guarded alongside
`/v1`, verified live, and the deploy no longer has a security blocker.

## The API surface is guarded, and the guard was verified against a real server

Leaving it open was the one thing that made a public deploy irresponsible: an
unauthenticated endpoint that accepts compile jobs. It is now guarded alongside
`/v1`, verified against a running server, and the deploy no longer has a security
blocker.

The guard was checked end to end rather than asserted:

```
GET  /health              -> 200          open by design
POST /v1/systems  (no key)-> 401          "missing or invalid API key"
POST /v1/systems  (key)   -> 422          past auth, validation as expected
POST /mcp         (no key)-> 401          was 200 before the guard
POST /mcp         (key)   -> 200, 8 tools
```

A full workflow, including the failure that proves the model host is per-request:

```
POST /v1/systems                    -> accepted
POST /v1/systems/:id/evals          -> approved, 2 train / 1 dev / 1 test
GET  /v1/systems/:id/report         -> 409 (phase error: not compiled yet)
POST /v1/systems/:id/compile        -> 503 "model server unreachable at
                                       https://model-host.invalid/v1/chat/completions"
```

That last line is the useful one: the error names **the base_url the caller passed**,
not the local default, so the per-request model host is genuinely honoured. A caller
who points it at a real server gets a compile; a caller who does not gets a message
that says so.

Two limitations, both demonstrated rather than predicted:

- **No model host ships with the API.** A container's `127.0.0.1` is the container, so
  the default model URL points at nothing. `MEKOY_MODEL_BASE_URL` sets a deployment
  default and a caller may override per request — which is what the 503 above proves.
  Compiling needs someone to point it at a reachable model server.
- **The store is process-local.** A restart lost a System created minutes earlier; the
  next request answered `system not found`. Persistent object storage would fix it and
  needs a hosted bucket, so what ships is the interface plus a local implementation.

## The model axis was missing, and searching it changed the ranking

A completion audit against the intended search surface — an ASHA controller over
{model, decode, k-shot} — found the search had **no model axis at all**.
`HarnessConfig` varied k-shot, retries, decode, prompt, schema, and bootstrap, but
never the base model. The model is named first in the
model first in its candidate pool and samples over three of them, so the search was
missing its largest lever and had been for the whole project.

Now that it exists, `--also-model` and one 33-minute search over two models:

| candidate | dev quality | latency |
|---|---|---|
| **7B k=4 r=0 schema strict** | **0.985** | **7.8s** |
| 14B k=0 r=0 free strict | 0.970 | 15.3s |
| 14B k=8 r=0 grammar default | 0.955 | 12.0s |
| 14B k=4 r=1 schema default | 0.929 | 11.8s |

Winner: the 7B, test quality 0.940, **7.8 s/doc**.

**The model axis lost to the harness axis, and the smaller model won on both quality
and speed.** Every 14B candidate was beaten by a 7B with the right harness, at roughly
half the latency. That is the whole thesis — *"optimize model + harness
together"*, and a specialist beating a bigger general model — measured on this task
rather than asserted.

It also retroactively explains the schema result. Isolated, schema-constrained decoding
scored slightly *worse* than `json_object` on Ollama (0.910 vs 0.925). Inside the
search it is part of the winning configuration, because it pairs with the strict
prompt at zero retries. One-arm comparisons miss interactions; that is what the
search is for.

## Does the eval hold up?

A benchmark you cannot audit is a benchmark you cannot trust, so labels are
checked against their own transcripts:

```
mekoy check-eval examples/bucko-restaurant/combined.jsonl
```

| corpus | rows | support their label |
|---|---|---|
| hand-labeled `examples.jsonl` | 12 | **12** |
| deterministic `generated.jsonl` | 96 | **96** |
| paraphrased `paraphrased.jsonl` | 175 | 174 |
| combined | 271 | 270 |

`label_problems()` rejects a row when the transcript contradicts its gold — it
claims a booking that did not happen, says nobody spoke on a row labeled
`unknown`, or never mentions the restaurant, party size, time, or name the label
asserts. The one remaining flag is a row where the note says "you wanted to
check on a reservation for three," which genuinely supports either intent.

**Building this tool falsified its own first version.** The initial check required
an explicit booking request for `reservation` and explicit inquiry language for
`availability`. Run against the hand-labeled fixture it failed four rows —
including *"host said a table for two is free tonight at 8, they did not book
it"*, which is plainly an availability call. Real transcripts imply the request.
The rule was wrong, not the data, so it was cut to the one case that is actually
a contradiction: an availability label on a transcript that asks to book.

Three more defects surfaced the same way, all found by auditing rather than
reasoning:

- `"I did not make a reservation"` was read as a booking request. Same negation
  bug already fixed for booking claims, missed in the second pattern.
- `"a table is free"` was not counted as availability language; only `open` and
  `available` were.
- `"staff never explicitly confirmed"` was not counted as unclear language.

Every one of those would have quietly mislabeled or mis-scored rows.

## What is honestly true

**We win on cost.** $0.00 per document against $0.017 — locally served, no
marginal price, no data leaving the machine. That is the durable win, and at a
few calls a week the absolute saving is small (about $4/year per active user of
this job). Cost is the weakest of the three claims commercially even though it is
the one we win outright.

**We win on quality against five of six frontier configurations, and against the
naive baseline.** Compiling adds +5.3 points over calling the same model one-shot
(0.902 → 0.955 on the 96-row corpus; the effect survives on the bigger one).
Two things are true at once and both should be said: on a 54-row test split the
compiled System is level with GPT-5.5 one-shot within five fields, and GPT-5.5
still edges it. The defensible claim is "level with frontier at zero marginal
cost", not "beats frontier".

**More shots helped the local model and hurt frontier.** `gpt-5.5` scored 0.958
one-shot and 0.937 with two shots; `claude-opus-4.8` 0.918 then 0.937. The
compiled System's dev-selected harness is k=8, so it is the one gaining from its
examples. A frontier API is not automatically better with more context.

**We do not win on speed.** 10.2 s/document locally against 1.4–1.7 s/call for a
frontier API, so local is roughly 6× slower. This was the opposite of the earlier
guess that local inference would be competitive, and it is measured, not assumed.
A batched or GPU-served runtime is the only path to a speed claim, and that is
unverified.

## Things we tried that did not work

- **Self-consistency voting.** Three samples at temperature 0.7, majority vote per
  closed field: dev quality identical (0.970 → 0.970) at 3.2× the latency
  (8.3 s → 26.9 s per document). The errors on `intent` and `status` are
  systematic, so the model makes the same mistake every sample and voting cannot
  reach it.
- **More shots, past a point.** On the 96-row corpus 8 shots scored *worse* on dev
  (0.955) than 2 (0.970). On the 271-row corpus the dev-selected winner is k=8.
  The right number of examples depends on the corpus, and neither extreme is
  universal.
- **Exact string comparison on free-text fields.** This is the bug that flattered
  the eval twice. First `"Saturday 9am"` was counted wrong against gold
  `"9am Saturday"`; then `"Sunday at 1:00 PM"` was counted wrong against
  `"Sunday at 1pm"`. Free text is judged, never gated on exact match. `when` now normalizes time formatting (glued meridiems,
  trailing `:00`, filler words) and only compares an AM/PM marker when both sides
  give one. `strict` is still reported alongside so the difference is not hidden.

## What remains unproven

- An eval this small cannot rank systems. One field is 0.071 of the score at
  n=19, and 0.036 at n=28. Treat every gap below ~0.04 as noise.
- The generated corpus is real but synthetic. It shares the author's blind spots
  about how a restaurant actually phrases a refusal.
- Only one open model family was tried, on one machine, on one task class.
- Cost is a list-price comparison, not a metered bill.

## Reproduce

```bash
uv run python -m mekoy.evalgen                 # regenerate the 96-row corpus
uv run pytest -q                               # 84 tests, no model server
mekoy eval --approve examples/bucko-restaurant/generated.jsonl
mekoy compile examples/bucko-restaurant/generated.jsonl --trials 6
mekoy bakeoff examples/bucko-restaurant/generated.jsonl \
  --baseline-model qwen2.5:7b --model qwen2.5:14b --trials 2
```

`--baseline-model` runs the baseline lane locally, so the whole bake-off works
with no API key and nothing leaving the machine. Omitting it uses a closed API,
and the baseline lane is the only place a closed API may touch this pipeline
(`src/mekoy/privacy.py` enforces that at the compile boundary).
