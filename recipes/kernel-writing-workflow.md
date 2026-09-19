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

When parallel work is useful, one coordination owner maintains hypotheses,
resource reservations, integration and acceptance; the lead can also implement.
Give each worker a falsifiable mechanism, private files, testbed identity,
deliverable, budget and stop criterion. Workers may refine the plan. Parallelize
independent variants; serialize shared edits, integration and final evaluation.

Assign distinct mechanisms, not just different tile values. Collect initial
hypotheses before exposing peers’ tentative winners and reserve bounded initial
screens for promising alternatives. Stop dead branches rather than maintaining
diversity for its own sake. Reserve runs in the task ledger, label intentional
replicas, and report at decisions/blockers with source, evidence and next action.
Worker timings are screens; evaluate frozen finalists under the common protocol.
Combine ideas only within authorized collaboration, subject to attribution rules.

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
