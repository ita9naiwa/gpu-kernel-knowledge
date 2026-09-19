# Kernel-writing workflow

These owner-authorized operating lessons include generalized nonpublic experience.
They are decision guidance, not public benchmark evidence or a proven optimal
agent policy. Apply the [independent-work rules](../SKILL.md#independent-optimization-and-attribution--mandatory).

## Startup and coordinated search

1. **Resolve the contract.** Inspect the caller/config and existing evidence first.
   Ask only unresolved questions affecting workload, forward/backward boundary,
   allowed transformations, accuracy or GPU/time budget. State reversible
   assumptions; continue independent inspection while a required answer is pending.
2. **Freeze a small testbed.** Reuse the harness or add only a missing adapter.
   Record source, effective precision/fusion/backend/compile/checkpoint/Graph
   settings, fixture/seed and the numerical gate in the [task record](../templates/task.md).
   Distinguish a precision-matched control from a quantized deployment target.
   Reuse applicable measurements or run baseline smoke and coarse profiling
   within budget. If profiling is unavailable, state the gap and choose a bounded
   diagnostic before ranking changes. A broken baseline is setup work, not a win.
   Harness corrections start a new comparison.
3. **Choose and screen a mechanism.** Rank attainable whole-boundary savings,
   set a minimum useful gain and cost cap, then use the [source navigation route](../sources/README.md#navigate-for-a-decision).
   Where scope permits, inspect the installed library path before replacing a
   dominant GEMM/attention operation; count wrapper, packing and backward costs.
   Preserve semantics and fallbacks in a bounded change with a cheap rejection check. Revisit the mechanism
   when evidence contradicts it rather than extending the same sweep blindly.
4. **Validate and retain.** Apply the timing protocol below and test the real
   consumer/output/gradient contract. A layer win is provisional, not a model
   speedup; do not sum unrelated ablation gains. Record live/peak allocated,
   reserved and device-used memory separately. Keep failures and scope limits
   in the [experiment record](../templates/experiment.md); re-profile after gains.

## Default team operation

Unless the user or task instructions specify otherwise, use these defaults for
an authorized cooperative optimization team. Counts include the lead. Stay with
one agent for trivial/sequential work or when no useful independent task exists;
otherwise start with two and expand only within available resources and budget.
These are operating defaults, not a proven optimal team size.

| Agents | Ownership | When to use |
| --- | --- | --- |
| 2 | Lead coordinates, evaluates and implements hypothesis A; worker owns hypothesis B. During setup, split baseline/profile from source/dispatch inspection. | Normal starting point for two useful independent directions. |
| 3 | Lead owns diagnosis, coordination and common evaluation; two workers own distinct hypotheses. | Evaluation/integration work would otherwise stall two substantial implementations. The lead investigates the next decision rather than waiting idle. |
| 4 | Lead coordinates/evaluates; two workers implement; the fourth owns a third plausible mechanism or independent consumer/backward/edge-case validation. | A third direction or validation bottleneck justifies the extra worker; do not invent work to fill a slot. |

Assign mechanisms such as data movement, dispatch or activation lifetime, not
merely languages or tile values unless those are the experiment's explicit target.
Each assignment states hypothesis, private files/worktree, frozen testbed,
deliverable, GPU/time budget and stop criterion. Workers may refine hypotheses;
collect initial proposals before exposing tentative winners, and give promising
alternatives their bounded initial screen before reallocating remaining budget.

Use the [task ledger](../templates/task.md) for ownership and run reservations.
Agent count does not imply GPU count: reserve slots before launch, serialize runs
when GPUs are scarce, and do CPU/source work while waiting. Label intentional
replication so it is distinguishable from duplicate work.

Report after a screen, blocker or contract/direction change, rather than every tool
call: hypothesis, source revision, result/evidence path, correctness, timing scope
and next decision. Escalate shared-code/resource conflicts before proceeding.
The lead summarizes accepted gains, rejected branches, blockers and the next
experiment to the user at milestones or the user's requested cadence.

Parallelize private variants; the lead serializes shared edits, integration and
final acceptance. Worker timings are screens. Evaluate frozen finalists with the
common protocol, integrate compatible changes one at a time and remeasure the
combination. Stop dead branches; useful negative evidence earns recognition, not
automatic extra budget. Apply the attribution rules: this team shares only within
authorized collaboration, never across independent with/without conditions.

## Set the numerical contract before search

Resolve the mode before implementation or broad parallel search. Reuse explicit
user requirements; otherwise state the default below and ask only material
unknowns. Do not infer permission for lossy arithmetic from a speed objective.

| Mode | Meaning | Required acceptance gate |
| --- | --- | --- |
| Default: ordinary numerical variation | Small errors from accumulation/reduction order, or measured nondeterminism already present in the baseline, are acceptable while preserving the operation and precision contract. | Set workload-specific absolute/relative error limits using dtype, scale, trusted-reference checks and repeated baseline runs. Baseline noise is evidence, not blanket permission for arbitrary errors. |
| Lossy, explicitly allowed | Approximate computation or precision changes may trade accuracy for speed within an agreed budget. | Ask for or agree to explicit cosine similarity minimum and relative L2 maximum, plus any application-level quality gate. Never invent a universal threshold. |
| Bitwise same | Exact equality with the agreed reference is required. | Match dtype and output bits for the specified outputs/state and inputs. If the reference is nondeterministic, resolve the deterministic reference/execution setup first; do not silently weaken the requirement. |

Freeze the reference, fixtures, dtypes, tensor scope, metrics and thresholds before
screening candidates; include gradients and the real consumer when relevant.
Define relative L2 as norm(candidate-reference)/norm(reference), including a
zero-reference policy. State cosine aggregation (per row/token/tensor), its
zero-vector policy, whether thresholds apply to worst cases or aggregates, and
whether reported L2 uses a fraction or percent. Reject unexpected NaN/Inf; aggregate
cosine alone must not hide a critical outlier or output/gradient failure.
Keep correctness checks during rough early timing. Relaxing a gate after failure
requires an explicit contract revision and reevaluation, not retroactive acceptance.

## Match benchmark effort to the decision

These are decision guidelines, not fixed statistical guarantees. Preserve the
correctness gate throughout; rough timing is not permission to skip validation
or compare different work. Warm compilation out of a warm-runtime measurement.

| Stage | Timing effort | Decision |
| --- | --- | --- |
| Early screening | Short warmup and a few trials, for example 3–5; consistent scope, input and Graph mode | Reject obvious regressions and retain large apparent gains provisionally. |
| Promising or close candidates | More repetitions with alternating order, raw samples and spread | Check whether the gain exceeds noise and reproduces. |
| Final result | Frozen common evaluator, matched hardware/inputs/mode, full declared correctness and replay checks | Report latency and memory; include required packing, saved-state or recomputation costs. |

Do not spend a final-validation budget measuring every early losing variant.
Conversely, do not call a small timing difference a win based on a few trials.
If short runs are unstable, increase measurement effort before screening.
For small final gains, predeclare the estimator, comparison order and stopping
rule. Use another device or input state when the claim needs that scope. Retain
failed confirmations and disclose sensitivity to order or aggregation; do not
repeat until a favorable interval appears. More trials cannot repair asymmetric
work, missing copies/recomputation or an incorrect reference.

Keep research-only tasks research-only: unavailable runtime measurements limit
the conclusion; they do not authorize inventing a benchmark result.

## Evidence-based feedback (experimental)

In-context feedback does not update model weights; its benefit is unproven here.
Prefer brief evaluator-verified explanations over numeric rewards. Credit a
validated gain, a material question that changes a decision, or a reproducible
rejection of an important plausible hypothesis. Do not credit question counts,
novelty, easy straw-man rejections or inconclusive failures alone.

Separate recognition from resource allocation: a useful rejection can deserve
credit while its branch stops. Allocate remaining budget by expected gain,
uncertainty and cost. Workers propose findings; the evaluator verifies them.

## Assess knowledge-assisted exploration

### Knowledge-use debug log

When enabled through the [skill](../SKILL.md#optional-knowledge-use-debug-log),
append a compact Markdown table to `knowledge-use.md` in the current task's
artifact directory, outside the installed/public KB. State its resolved path
once. Record the KB revision and when tracing began; do not reconstruct earlier
reads as contemporaneous events. Reuse an existing task log if it has these fields.

| Event / time / agent | Stage or attempt | Question or trigger | Knowledge actually read | Use and observable decision |
| --- | --- | --- | --- | --- |

Append an event immediately after a targeted read or related decision, before
the next implementation/experiment. Use a sequential event ID and an observed
timestamp with timezone when available; otherwise use the event order without
inventing a time. Identify the document path and section/rule ID, plus source
revision or pinned URL for upstream material. Distinguish a search hit or failed
retrieval from a section actually read; a linked source is not automatically read.

Record whether the material was applied, ruled out, confirmed the existing plan,
or left unresolved, with a short reason tied to workload conditions and the next
check. For a later decision or result, append a follow-up referencing the event
and experiment/evidence path rather than rewriting the original entry. Record
decisions and evidence, not private chain-of-thought or copied source passages.

Keep entries brief; do not add searches merely to populate the log. In a team,
workers keep separate logs identified by agent and the lead links them, avoiding
concurrent writes. Logs remain within the authorized task; never share them
between independent comparison conditions or publish private task details to
this KB. At completion, link the log and note any recording gaps. This is
agent-reported provenance, not exhaustive access telemetry or proof that a
reference caused a performance gain.

### Comparison criteria

Distinguish proposed mechanisms, executed experiments, passing variants and
adopted whole-boundary gains. Compare time to diagnosis, accepted gains per
GPU/time budget, useful coverage, duplicated work and unnecessary questions,
including coordination cost. Breadth or a longer bibliography alone is not success.

If testing feedback, compare no added feedback against evidence-based feedback
on multiple matched-budget tasks. Freeze access, testbeds and judging rules;
retain failures and revise or remove feedback based on results. Separate phases
when resources, knowledge access, baseline or numerical gates change. A common
evaluator must preserve contestant independence. These recommendations authorize
no additional experiments and establish no causal benefit by themselves.
