# RSGA MACE Optimized

This repository contains the optimized MACE-integrated Reciprocal Space Gated
Attention codebase, referred to as MACERSGA. It corresponds to the stable
float64 production code used for the 2026 In2Se3 molecular
dynamics runs on CCR.

The package is stored under `mace/`, following the layout of the published
[`GSLab2025/MACE_RSGA`](https://github.com/GSLab2025/MACE_RSGA) repository.

## Hardware Scope

The optimized MACERSGA code is implemented through PyTorch tensor operations
and does not require a custom NVIDIA-only CUDA extension for correctness. The
main optimizations--shared per-forward RSGA geometry context, reciprocal-grid
caching, chunked large-graph evaluation, strict dtype controls, and robust
`torch.compile` fallback behavior--are backend-portable in principle and should
run on CPU or other PyTorch-supported accelerators when the dependencies are
available.

The production performance path was designed, tested, and validated on NVIDIA
CUDA GPUs, specifically A100 and H100-class hardware, in strict float64 mode.
The documented defaults disable TF32 and optional fp32 fast-eval paths to
preserve long-range physics. CPU, AMD GPU, Apple silicon, or other non-CUDA
backends should be treated as portable but not performance-validated; users
should benchmark and numerically validate those platforms before production MD
or training.

## Performance Gain

"Optimized" here refers to reducing the RS-GA overhead in large periodic
systems while preserving the strict float64 physics path. The main changes are
shared per-forward geometry context across RS-GA layers, reciprocal-grid
caching for repeated cells, chunked large-graph evaluation to control peak
memory, and a guarded large-system path that removes the dominant batched
pairwise tensor contraction hotspot.

Observed validation benchmarks against the earlier MACERSGA implementation:

| System and path | Previous implementation | Optimized implementation | Change |
| --- | ---: | ---: | ---: |
| 3000-atom silica forward/eval wall time | 1.220139 s | 0.526204 s | 2.32x faster |
| 3000-atom silica peak GPU memory | 16999 MB | 7619 MB | 2.23x lower |
| 3000-atom silica `aten::bmm` self device time | 835.313 ms | 23.391 ms | 35.7x lower |
| 240-atom In2Se3 forward/eval wall time | 0.200606 s | 0.200651 s | essentially unchanged |

The 240-atom case is intentionally flat: the large-graph path is guarded so
small systems do not pay overhead for an optimization meant for large cells.

Expected gain: speedup is most likely for large enough systems, roughly when
the RS-GA pairwise/batched-matmul path becomes a major cost and the calculation
is close to a GPU memory limit; small cells, MACE-dominated workloads, or runs
with very few reciprocal modes may show little or no speedup.

## Installation

Create or activate the intended conda environment, then install the package
editable:

```bash
python -m pip install --no-deps -e mace/
```

The validated strict float64 runtime settings are:

```bash
export MACE_RSGA_ALLOW_TF32=0
export MACE_RSGA_FAST_EVAL_FP32=0
export MACE_RSGA_CHUNK_MB=1024
```

`MACE_RSGA_ALLOW_TF32=0` disables TF32 matmul paths for strict long-range
physics.

`MACE_RSGA_FAST_EVAL_FP32=0` disables optional fp32 inference shortcuts.

`MACE_RSGA_CHUNK_MB=1024` controls the memory target for chunked large-graph
RSGA evaluation.

If a runtime has a `torch.compile` issue in the reciprocal-cell helper, set:

```bash
export MACE_RSGA_DISABLE_TORCH_COMPILE=1
```

The default behavior tries the compiled helper first and automatically falls
back to eager execution if the backend fails.

## Production Notes

- The RS-GA block is integrated as a layerwise embedding correction to the
  short-range MACE stack.
- The optimized code preserves the node-level SR/LR mixing gate used in the
  production physics model.
- ZBL handling is not scale-shifted by the learned energy scale/shift path.
- The production path keeps TF32 and fp32 fast-eval disabled for float64
  long-range physics.

## How to use ?

Always include the pair repulsion. Example entry point in the modified MACE stack:

```bash
mace_run_train --model="MACERSGA" --pair_repulsion  --distance_transform="Agnesi"  ...
```

## Provenance

This code descends from:

- [`GSLab2025/RSGA`](https://github.com/GSLab2025/RSGA)
- [`GSLab2025/MACE_RSGA`](https://github.com/GSLab2025/MACE_RSGA)

The standalone optimized RSGA modules are maintained in:

- [`GSLab2025/RSGA_OPTIMIZED`](https://github.com/GSLab2025/RSGA_OPTIMIZED)
