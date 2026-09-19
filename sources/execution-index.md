# Model execution index

Use this shared route map before searching upstream trees. Scope: one model's
forward, backward, optimizer, state memory and multi-GPU/node communication.
Request routing, dynamic request batching, replicas and autoscaling are excluded;
pipeline-parallel microbatch schedules are included.

## Reuse the map

1. Match the bottleneck to a row; read only its relevant entry files first.
2. Resolve source IDs through [catalog.json](catalog.json): pins, inspected paths,
   licenses and limits live there. Local sources are `upstream/<source-id>/`
   relative to the skill root (`pytorch-inductor` uses `upstream/pytorch/`).
3. If a route misses, run targeted `rg` in that source. Save a useful new entry
   once in the catalog and amend the affected row; do not rebuild a full index
   for each task or agent. A coordinator integrates concurrent discoveries.
4. On a source upgrade, check only affected paths/citations against the new pin.
   A prepared checkout is not evidence of reading; a source read is not a test.

## Question → source → entry

The links below pin inspected entry points. Read the installed caller/API first:
these research snapshots need not match your target environment.

| Question / bottleneck | Sources and initial entry files | Transfer boundary |
| --- | --- | --- |
| Video DiT execution / sequence parallelism | `fastvideo`: [fastvideo/models/wan/transformer.py](https://github.com/hao-ai-lab/FastVideo/blob/430e52154e76b902c3cc17a16b3edc1fad790012/fastvideo/models/wan/transformer.py)<br>`fastvideo`: [fastvideo/distributed/parallel_state.py](https://github.com/hao-ai-lab/FastVideo/blob/430e52154e76b902c3cc17a16b3edc1fad790012/fastvideo/distributed/parallel_state.py) | Check sequence partitioning, attention backend and process-group topology. |
| Diffusion attention backend / memory offload | `diffusers`: [src/diffusers/models/attention_dispatch.py](https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/src/diffusers/models/attention_dispatch.py)<br>`diffusers`: [src/diffusers/hooks/group_offloading.py](https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/src/diffusers/hooks/group_offloading.py) | Check tensor contracts, streams and offload synchronization; model code, not request management. |
| Training step / recomputation / optimizer | `torchtitan`: [torchtitan/training_engine.py](https://github.com/pytorch/torchtitan/blob/6c2dadbb3109998092d1164e990c6377b2b8fb50/torchtitan/training_engine.py)<br>`torchtitan`: [torchtitan/distributed/activation_checkpoint.py](https://github.com/pytorch/torchtitan/blob/6c2dadbb3109998092d1164e990c6377b2b8fb50/torchtitan/distributed/activation_checkpoint.py) | Match autograd, parallel dimensions and activation lifetime; optimizer entries are in the catalog. |
| Tensor / pipeline parallelism and overlap | `megatron-lm`: [megatron/core/tensor_parallel/layers.py](https://github.com/NVIDIA/Megatron-LM/blob/e22d5b955dcb6f43c138625437e4af0865d28983/megatron/core/tensor_parallel/layers.py)<br>`megatron-lm`: [megatron/core/pipeline_parallel/schedules.py](https://github.com/NVIDIA/Megatron-LM/blob/e22d5b955dcb6f43c138625437e4af0865d28983/megatron/core/pipeline_parallel/schedules.py) | Match forward/backward collectives, microbatch schedule and numerical contract. |
| Optimizer state / ZeRO memory | `deepspeed`: [deepspeed/runtime/engine.py](https://github.com/deepspeedai/DeepSpeed/blob/215f5aa9c4051d2d64fa71753e8d8431c1c9b1e2/deepspeed/runtime/engine.py)<br>`deepspeed`: [deepspeed/runtime/zero/stage3.py](https://github.com/deepspeedai/DeepSpeed/blob/215f5aa9c4051d2d64fa71753e8d8431c1c9b1e2/deepspeed/runtime/zero/stage3.py) | Start at engine and zero/stage3; include communication and offload cost in model timing. |
| Inference engine / CUDA Graph execution | `tensorrt`: [samples/common/sampleInference.cpp](https://github.com/NVIDIA/TensorRT/blob/c93b7d4893184af4882f9f1862e13a5c64b8677d/samples/common/sampleInference.cpp)<br>`tensorrt`: [samples/trtexec/README.md](https://github.com/NVIDIA/TensorRT/blob/c93b7d4893184af4882f9f1862e13a5c64b8677d/samples/trtexec/README.md) | OSS samples and public APIs; proprietary engine internals are not available here. |
| Quantized distributed model inference | `tensorrt-llm`: [tensorrt_llm/_torch/modules/linear.py](https://github.com/NVIDIA/TensorRT-LLM/blob/5958f7c70046a996461fa001ca83363708a28913/tensorrt_llm/_torch/modules/linear.py)<br>`tensorrt-llm`: [tensorrt_llm/_torch/distributed/ops.py](https://github.com/NVIDIA/TensorRT-LLM/blob/5958f7c70046a996461fa001ca83363708a28913/tensorrt_llm/_torch/distributed/ops.py) | Inference paths do not establish backward support; exclude service orchestration. |
| MoE dispatch / combine | `deepep`: [deep_ep/buffers/elastic.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/buffers/elastic.py)<br>`deepep`: [tests/elastic/test_ep.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/elastic/test_ep.py) | V2 ElasticBuffer uses NCCL Gin (NCCL >=2.30.4); legacy NVSHMEM paths differ. Check events and buffer ownership. |
| Collective transport / GPU-side collectives | `nccl`: [src/transport.cc](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/transport.cc)<br>`nccl`: [docs/examples/06_device_api/01_allreduce_lsa/README.md](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/docs/examples/06_device_api/01_allreduce_lsa/README.md) | Check rank topology, symmetric windows and collective barrier participation; use matching collective tests. |
| One-sided communication / compute overlap | `nvshmem`: [examples/putmem_signal_counted.cu](https://github.com/NVIDIA/nvshmem/blob/b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2/examples/putmem_signal_counted.cu)<br>`nvshmem`: [examples/gemm_allreduce/gemmAR_fusion_blackwell_fp16.cu](https://github.com/NVIDIA/nvshmem/blob/b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2/examples/gemm_allreduce/gemmAR_fusion_blackwell_fp16.cu) | Examples have build and hardware constraints: counted signals need CFT_HANDLES; Blackwell GEMM example needs CUTLASS. |

For existing kernel routes, use the [source map](README.md) and
[validation map](validation-map.md): **DeepGEMM** for grouped/quantized GEMM;
**PyTorch** for autograd, operators and Inductor; **vLLM/SGLang** for model
layers, backend dispatch, quantization and graph integration. Restrict their
serving-heavy trees to the model execution issue being investigated.

## Evidence / maintenance

Added routes inspected on 2026-09-19; older records retain their original audit
dates. The catalog records source sections, not exhaustive file reviews. No new
GPU, distributed correctness or performance validation is implied. Before reuse,
select a matching upstream test (where available), then validate in the target
model with its agreed tolerance and timing contract. See [audit](audit.md).

This is a curated entry index, not a symbol database. Add deeper indexing only
when repeated targeted misses justify it; do not scan all repositories up front.
