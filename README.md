# RSGA MACE Optimized

This repository contains the optimized MACE-integrated Reciprocal Space Gated
Attention codebase, referred to as MACERSGA. It corresponds to the stable
float64 production code used for the 2026 In2Se3 4x4x1 and 9x9x1 molecular
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


## Provenance

This code descends from:

- [`GSLab2025/RSGA`](https://github.com/GSLab2025/RSGA)
- [`GSLab2025/MACE_RSGA`](https://github.com/GSLab2025/MACE_RSGA)

The standalone optimized RSGA modules are maintained in:

- [`GSLab2025/RSGA_OPTIMIZED`](https://github.com/GSLab2025/RSGA_OPTIMIZED)
