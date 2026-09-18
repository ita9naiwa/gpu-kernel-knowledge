# Architecture selection: target and instruction constraints

Tags: Blackwell, B200, B300, RTX 5090, SM100, SM103, SM120, ISA, dispatch, compatibility.
Evidence: official architecture/ISA documentation and pinned CUTLASS documentation.
The workflow below is a proposed validation sequence, not a tested port.

## First decision: what GPU is actually executing?

Do not use a marketing family as an instruction-support predicate. NVIDIA's
[compute-capability table](https://developer.nvidia.com/cuda/gpus), checked
2026-09-18, lists H100/H200 as 9.0, B200/GB200 as 10.0, B300/GB300 as 10.3,
and RTX 5090/RTX PRO 6000 Blackwell as 12.0. Record the runtime device, compiler
target and toolkit separately. A source example named `sm100` is not a claim
about every GPU with Blackwell in its name.

| Layer | Required evidence | Failure if skipped |
| --- | --- | --- |
| Device | Runtime GPU and compute capability | Code selected for a different architecture |
| Instruction | Exact PTX operation, qualifiers and Target ISA Notes | Unsupported instruction or invalid operand mode |
| Compiler/build | Toolkit, library revision and emitted targets | Source supports a feature but the installed binary does not |
| Runtime | Backend eligibility and executed kernel | New code compiles but the application still uses a fallback |
| Workload | Shape/layout/dtype and full timing scope | Legal kernel is slower or semantically incompatible |

## A concrete reason the exact instruction matters

The inspected [PTX ISA 9.4 `tcgen05.mma` target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma)
list `.kind::i8` for `sm_100a`, the historical `sm_101a` spelling, and `sm_110a`;
they do not grant that qualifier to `sm_103a` merely because other `tcgen05`
operations support it. This is a narrow statement about this instruction kind,
not a statement that B300 cannot perform any integer arithmetic or any INT8
matrix multiplication. Inspect an alternative path rather than extrapolating
from the datatype or product name.

Similarly, the [CUTLASS Blackwell functionality document](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/blackwell_functionality.md)
explicitly scopes its opening instruction table to SM100. Its datatype support
table cannot establish an SM103 build's actual emitted kernel coverage. The
source generator, compile-time guards and runtime dispatch are further evidence
needed for an installed application.

## Resource limits are another contract

The [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)
gives different resident-warp and shared-memory limits for compute capabilities
10.0 and 12.0. A tile/stage choice fitting one can exceed the other's per-block
budget. Use device attributes and compiled resource usage rather than copying a
shared-memory number across the family. For clusters, inspect active-cluster
occupancy and launch constraints, not only ordinary block occupancy.

Do not maximize occupancy as an end in itself. A smaller tile can increase
resident blocks while creating more memory traffic or synchronization. Keep
resource measurements tied to the observed latency and bottleneck hypothesis.

## Binary portability is not performance portability

The [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html#application-compatibility-on-blackwell-architecture)
distinguishes ordinary PTX forward compatibility from architecture-conditional
targets such as `compute_90a`/`compute_100a`, which have narrower compatibility.
Inspect the actual embedded cubins/PTX and installed toolchain. A fallback JIT
path that launches successfully is not evidence that a new tensor-core mechanism
is being used or that steady-state performance is optimal.

## Bounded validation sequence

1. Save actual device/toolchain/library versions and confirm the intended target
   exists in the installed compiler and library.
2. Trace the chosen operation through its dtype/layout/shape/architecture guards;
   identify the fallback and inspect compile output for the intended variant.
3. Run correctness and safety checks on the smallest representative workload,
   including a case that intentionally exercises the fallback.
4. Profile a representative launch to identify the kernel and resource usage;
   measure performance separately without profiler perturbation.
5. Compare the same workload on each required GPU target. Record unsupported and
   untested combinations instead of filling a support table by assumption.

These architecture-selection claims were not verified by GPU execution. Recheck the rolling NVIDIA
pages against the compiler version you actually deploy.
