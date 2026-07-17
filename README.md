# SIMD Matrix-Vector Multiply Kernel (x86-64 NASM)

A hand-written matrix-vector multiply (GEMV) kernel in raw x86-64 assembly, built in two versions: a naive scalar baseline and an AVX2 + FMA vectorized version. No intrinsics, no compiler autovectorization. Both kernels are called directly from C using the System V AMD64 calling convention and benchmarked against each other on real hardware.

GEMV is not a random choice of demo. It is the operation that dominates the decode step of transformer inference. During autoregressive generation, batch size collapses to one token at a time, so every QKV projection and every FFN layer becomes a matrix-vector product, not a matrix-matrix product. Understanding why this operation is slow and how to speed it up is directly relevant to inference latency work in frameworks like llama.cpp and ggml.

## Problem

A matrix-vector multiply is O(rows x cols) scalar multiply-adds. Written naively, the CPU processes one float at a time, using at most 1/8th of the floating point throughput available on any machine with AVX2. The question is how much of that gap is closeable by hand, and where the ceiling actually is.

## Technical decisions

**AVX2 + FMA over AVX-512.** The target CPU supports AVX-512, but AVX2 is the realistic baseline for laptop and commodity server hardware in 2026. Optimizing for hardware most people don't have isn't a useful demonstration.

**Fused multiply-add (`vfmadd231ps`) instead of separate multiply and add.** One instruction, one rounding step, half the register pressure of doing `vmulps` then `vaddps`.

**8-wide horizontal reduction, not a running scalar sum.** The inner loop accumulates 8 partial sums in a single `ymm` register across the whole row, and only collapses to one scalar value once, at the end of the row, using `vextractf128` + `vaddps` + two `vhaddps`. Reducing on every iteration would erase the benefit of vectorizing in the first place.

**A separate scalar tail loop for remainder columns.** Real matrices are not always column counts divisible by 8. The kernel masks the vectorizable region with `and r10, -8`, runs the AVX2 loop up to that bound, then finishes the remaining 0-7 columns with scalar VEX-encoded instructions (`vmovss`/`vmulss`/`vaddss`, not legacy SSE, to avoid an AVX-SSE transition penalty on the same register file). This is the detail that makes the kernel correct on arbitrary input rather than only on conveniently-sized demo matrices.

**Correctness checked against a double-precision C reference**, not just eyeballed. Both assembly kernels are compared against a reference that accumulates in `double` to catch float32 accumulation drift, across four shapes including a non-multiple-of-8 column count (37) and a degenerate 1x1 case.

## Outcome

Measured on a GitHub Codespaces cloud VM (AVX2 + FMA capable), averaged over hundreds to thousands of iterations per size. Shared cloud hardware introduces some noise — note the dip at 256x256 — but the compute-bound-to-bandwidth-bound trend across sizes still holds:

| rows | cols | scalar (ns) | SIMD (ns) | speedup | scalar GFLOP/s | SIMD GFLOP/s |
|------|------|-------------|-----------|---------|----------------|--------------|
| 64   | 64   | 11,166      | 961       | 11.62x  | 0.73           | 8.53         |
| 128  | 128  | 18,967      | 1,765     | 10.74x  | 1.73           | 18.56        |
| 256  | 256  | 78,959      | 16,196    | 4.88x   | 1.66           | 8.09         |
| 512  | 512  | 250,315     | 31,407    | 7.97x   | 2.09           | 16.69        |
| 1024 | 1024 | 1,271,188   | 147,449   | 8.62x   | 1.65           | 14.22        |
| 2048 | 2048 | 4,692,473   | 617,945   | 7.59x   | 1.79           | 13.58        |
| 4096 | 4096 | 28,776,495  | 4,429,443 | 6.50x   | 1.17           | 7.58         |
| 4096 | 1000 | 4,187,663   | 566,053   | 7.40x   | 1.96           | 14.47        |

The honest finding is not "8x speedup everywhere," it's that speedup shrinks as matrix size grows: 11x at 64x64 down to 4.3x at 4096x4096. Small matrices stay resident in L1/L2 cache, so the SIMD kernel is genuinely compute-bound and gets close to its theoretical 8x width. Once the matrix stops fitting in cache, both kernels become memory-bandwidth bound: the CPU spends more time waiting on loads from main memory than executing arithmetic, so vectorizing the arithmetic has less left to speed up. This is the same reason GEMV during LLM decode is described as memory-bandwidth bound rather than compute-bound in inference engineering discussions, not a coincidence particular to this kernel.

Raw numbers: `results/benchmark_results.csv`.

## Project layout

```
src/matvec_scalar.asm   naive scalar baseline
src/matvec_simd.asm     AVX2 + FMA vectorized kernel, scalar tail for remainder columns
bench/main.c            correctness check (vs. double-accumulated C reference) + benchmark harness
Makefile                nasm -f elf64, links against the C harness
results/                benchmark_results.csv generated on run
```

## Build and run

```
make
./matvec_bench
```

Requires `nasm` and `gcc` on an x86-64 Linux machine with AVX2 and FMA support.
