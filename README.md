# SIMD Matrix-Vector Multiply Kernel (x86-64 NASM)

![NASM](https://img.shields.io/badge/NASM-x86--64-blue)
![AVX2](https://img.shields.io/badge/AVX2-FMA-orange)
![C](https://img.shields.io/badge/harness-C-informational)

A hand-written matrix-vector multiply (GEMV) kernel in raw x86-64 assembly, in two versions: a naive scalar baseline and an AVX2 + FMA vectorized version. No intrinsics, no compiler autovectorization. Both are called directly from C using the System V AMD64 calling convention and benchmarked against each other.

> **TL;DR:** up to **11.6x** faster than scalar on cache-resident matrices, settling around **6.5–8x** once the matrix outgrows cache — a real, measured demonstration of the compute-bound → bandwidth-bound wall.

I picked GEMV specifically because it's the operation that dominates the decode step of transformer inference. During token-by-token generation, batch size drops to one, so every QKV projection and every FFN layer becomes a matrix-vector product instead of a matrix-matrix product. Understanding why that operation is slow, and how much of it can be sped up by hand, is directly relevant to inference-latency work in engines like llama.cpp and ggml.

## Problem

A matrix-vector multiply is O(rows × cols) scalar multiply-adds. Done naively, one float is processed at a time — at most 1/8th of the floating-point throughput available on any AVX2-capable CPU. The question I wanted to answer: how much of that gap can actually be closed by hand, and where's the real ceiling.

## Technical decisions

- **AVX2 + FMA over AVX-512.** The dev machine supports AVX-512, but AVX2 is the realistic baseline for commodity laptop/server hardware. Optimizing for instruction sets most machines don't have isn't a useful exercise.
- **Fused multiply-add (`vfmadd231ps`) instead of separate multiply and add.** One instruction, one rounding step, half the register pressure of `vmulps` followed by `vaddps`.
- **8-wide accumulation, single horizontal reduction per row.** The inner loop keeps 8 running partial sums in one `ymm` register across the whole row, and only collapses them to a single scalar once — at the end of the row — using `vextractf128` + `vaddps` + two `vhaddps`. Reducing on every iteration would cancel out the point of vectorizing.
- **Scalar tail loop for remainder columns.** Column counts aren't always multiples of 8. The kernel masks the vectorizable region with `and r10, -8`, runs the AVX2 loop up to that bound, then handles the leftover 0–7 columns with VEX-encoded scalar instructions (`vmovss`/`vmulss`/`vaddss`) rather than legacy SSE, to avoid an AVX-SSE transition penalty on the same register file. This is what makes the kernel correct on arbitrary input sizes, not just convenient ones.
- **Correctness checked against a double-precision C reference**, not eyeballed. Both kernels are compared against a reference that accumulates in `double` to catch float32 accumulation drift, across four shapes including a non-multiple-of-8 column count (37) and a degenerate 1×1 case.

## Results

Measured on a GitHub Codespaces VM (AVX2 + FMA capable), averaged over hundreds to thousands of iterations per size. Shared cloud hardware adds some noise — the 256×256 dip is likely that — but the overall compute-bound → bandwidth-bound trend holds:

| rows | cols | scalar (ns) | SIMD (ns) | speedup | scalar GFLOP/s | SIMD GFLOP/s |
|------|------|-------------|-----------|---------|----------------|--------------|
| 64   | 64   | 11,166      | 961       | **11.62x** | 0.73  | 8.53  |
| 128  | 128  | 18,967      | 1,765     | **10.74x** | 1.73  | 18.56 |
| 256  | 256  | 78,959      | 16,196    | 4.88x      | 1.66  | 8.09  |
| 512  | 512  | 250,315     | 31,407    | 7.97x      | 2.09  | 16.69 |
| 1024 | 1024 | 1,271,188   | 147,449   | 8.62x      | 1.65  | 14.22 |
| 2048 | 2048 | 4,692,473   | 617,945   | 7.59x      | 1.79  | 13.58 |
| 4096 | 4096 | 28,776,495  | 4,429,443 | 6.50x      | 1.17  | 7.58  |
| 4096 | 1000 | 4,187,663   | 566,053   | 7.40x      | 1.96  | 14.47 |

The honest finding isn't "8x speedup everywhere" — it's that speedup shrinks as matrix size grows, from 11x at 64×64 down to roughly 6.5x at 4096×4096. Small matrices stay resident in L1/L2 cache, so the SIMD kernel is genuinely compute-bound and gets close to its theoretical 8x ceiling. Once the matrix stops fitting in cache, both kernels become memory-bandwidth bound — the CPU spends more time waiting on loads than doing arithmetic, so vectorizing the arithmetic has less left to speed up. This matches why GEMV during LLM decode is generally described as memory-bandwidth bound rather than compute-bound, not a coincidence specific to this kernel.

Raw numbers: [`results/benchmark_results.csv`](results/benchmark_results.csv)

## Project layout

```
src/matvec_scalar.asm   naive scalar baseline
src/matvec_simd.asm     AVX2 + FMA vectorized kernel, scalar tail for remainder columns
bench/main.c            correctness check (vs. double-accumulated C reference) + benchmark harness
Makefile                nasm -f elf64, links against the C harness
results/                benchmark_results.csv, generated on run
```

## Build and run

```bash
make
./matvec_bench
```

Requires `nasm` and `gcc` on an x86-64 Linux machine with AVX2 and FMA support.
