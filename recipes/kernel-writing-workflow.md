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
