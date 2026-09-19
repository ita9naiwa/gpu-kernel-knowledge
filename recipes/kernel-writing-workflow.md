# Kernel-writing workflow

This is a decision procedure, not measured evidence of a speedup. Follow the
[skill's optimization priorities and attribution rules](../SKILL.md).

1. **Establish the contract and limiting cost.** Recover the workload and freeze
   the agreed baseline using the [task record](../templates/task.md). Reuse
   applicable measurements or obtain a coarse breakdown at the requested timing
   boundary. Use [bottleneck triage](bottleneck-triage.md) to distinguish an
   observation from a cause; state unknowns if measurement is blocked.
2. **Consult the KB before choosing the change.** Select up to three relevant
   sections from the [practice map](../practices/README.md), read their guards
   and counterconditions, and follow the evidence needed for this decision.
   Check relevant [failure lessons](lessons-from-failed-attempts.md). Rank
   candidates by attainable time saved at the requested boundary. Record which
   guidance changed the choice and why it applies; use targeted primary-source
   research when the KB has a gap.
3. **Screen one hypothesis cheaply.** Record the mechanism, countercase and
   rejection check in the [experiment record](../templates/experiment.md).
   Implement the current task's own change, preserve fallbacks, and check
   correctness before short warmed timing runs. Revisit the relevant KB entry
   if parity fails or timing contradicts the mechanism, before adding changes.
4. **Validate the decision at the real boundary.** Use
   [measurement guidance](measurement-boundaries.md) for comparable execution
   modes and inputs. Increase repetitions and use matched alternating runs for
   small differences, selection and final claims. Retain unfavorable cases and
   separate kernel timing from application timing.
5. **Retain the result and choose the next step.** Record acceptance or rejection,
   evidence limits and useful failure conditions. Re-profile after a substantial
   gain, then retrieve for the new bottleneck. Report only verified incremental
   gains as new results; identify the KB guidance actually used and unrun checks.

## Startup and coordinated search

These generalized, owner-authorized operating lessons incorporate nonpublic
experience; private artifacts are omitted. They are recommendations, not public
benchmark evidence or a demonstrated causal benefit from this skill.

This is an operating recommendation, not a proven optimal agent count or budget
allocation. Use it when the task benefits from parallel hypotheses; simple tasks
stay with one agent.

1. **Resolve the contract with evidence and focused questions.** Inspect the
   actual caller/config and prior artifacts. If unresolved, ask which workload
   and F/B boundary matter, which output tolerance or exactness is required,
   whether library/precision/checkpoint changes are allowed, and what GPU/time
   budget is available. Bundle only material unknowns; do not ask again for facts
   already supplied. Do not silently relax semantics. Keep dependent experiments
   pending when an answer is required, while continuing independent inspection.
2. **Freeze a small runnable testbed.** Reuse the project's launcher and harness;
   write only the missing adapter/check. Pin baseline source and effective
   options, representative fixture/seed, output and gradient gates, and timing
   mode. Execute baseline smoke/replay and a coarse timing/profile within the
   authorized resources. A broken baseline is a setup task, not a candidate win.
   Save commands/results in the [task contract](../templates/task.md). Later
   harness corrections require a new comparison for affected candidates.
3. **Assign one coordination owner.** The lead agent owns the hypothesis list,
   expected impact, budgets, experiment reservations, integration and acceptance.
   It can also do useful local work; a dedicated idle manager is unnecessary.
   Workers propose/refine hypotheses rather than merely obeying the first plan.
   Give each a falsifiable mechanism, owned files, deliverable, testbed identity,
   GPU slot, cost cap and stop criterion. Parallelize independent variants in
   private files/worktrees. Serialize shared-source edits, integration and final
   acceptance through the owner; use an independent review pass when warranted.
4. **Reward useful coverage, not only the fastest candidate.** Initially assign
   distinct plausible mechanisms, such as traffic/layout, activation lifetime,
   and library dispatch. Tile variants alone are not independent directions.
   Ask workers for an initial hypothesis before showing peers' tentative winners.
   Reserve a bounded first screening opportunity for each promising direction;
   one early winner must not consume the entire budget. Give credit for either
   a validated whole-boundary gain or a decisive, reproducible rejection that
   closes an important uncertainty. Novelty, experiment count and inconclusive
   failures alone earn no credit. Reassign remaining budget by expected impact,
   evidence and cost; do not keep a dead direction alive to satisfy diversity.
5. **Share compact state at decision points.** Before a GPU run, reserve its
   hypothesis/owner/source/slot in a shared ledger; label intentional replicas.
   Report after a screen, blocker or contract change, not every tool call.
   Return source hashes, result path, correctness, timing scope, rejected
   alternatives and one next decision. Treat workers' timings as screens;
   evaluate frozen finalists under the common protocol. Preserve scoped failure
   records to avoid unchanged retries. Within an authorized cooperative team,
   combine compatible validated ideas after attribution checks; do not share
   solutions across independent with/without conditions.

A compact assignment can be: "Test whether [mechanism] removes [measured cost]
within [boundary]. Own [files] using [baseline/evaluator]. Budget [time/GPU].
Return [patch + result + counterexample]. Stop at [threshold/failure condition].
Coordinate before changing direction or shared code."

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
Measure repeated baseline behavior on representative and difficult cases, but do
not treat observed variability as proof that all deviations within it are harmless.
Keep correctness checks during rough early timing. Relaxing a gate after failure
requires an explicit contract revision and reevaluation, not retroactive acceptance.

## Evidence-based feedback (experimental)

Feedback in an agent's context does not update model weights. Its effect on
search quality is unproven here; do not make numeric rewards a default objective.
Start with short, specific feedback from the coordination owner or evaluator:

- Credit an answer or implementation for a verified improvement at the agreed
  boundary, not a persuasive explanation or self-reported score.
- Credit a question when it resolves a material ambiguity and changes a decision
  or prevents invalid work; do not reward question count or repeated clarification.
- Credit a rejection when reproducible evidence rules out a plausible, important
  hypothesis within a stated scope. Easy straw-man rejections, novelty, experiment
  counts and inconclusive failures are not contributions by themselves.

Explain the evidence and decision affected. Keep recognition separate from future
GPU/time allocation: a useful negative result may deserve credit while its branch
should stop. Allocate remaining resources by expected gain, uncertainty and cost.
Workers propose findings; the common evaluator verifies them and preserves failures.

If evaluating this policy, compare no added feedback with evidence-based feedback
across multiple matched-budget tasks. Freeze testbeds, judging rules and access;
measure accepted gains, useful hypothesis coverage, duplicate work and unnecessary
questions, including coordination cost. Adopt, revise or remove feedback based on
those results. This recommendation does not authorize additional experiments.

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

## Assess knowledge-assisted exploration

When comparing exploration methods, distinguish proposed mechanisms, executed
experiments, correctness-passing variants and adopted whole-boundary gains.
Record time to a useful bottleneck diagnosis, time/GPU budget to accepted gains,
and why losing branches were stopped. A broader search or longer bibliography
alone is not success; retrieval should change a concrete experimental decision.
Use an authorized common evaluator without sharing contestants’ solutions.
Separate phases when knowledge access, resources, baseline or numerical gates
change; unmatched final latencies do not establish a causal knowledge benefit.
