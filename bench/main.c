#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern void matvec_scalar(const float *mat, const float *vec, float *out,
                           long rows, long cols);
extern void matvec_simd(const float *mat, const float *vec, float *out,
                         long rows, long cols);

static void matvec_ref(const float *mat, const float *vec, float *out,
                        long rows, long cols) {
    for (long i = 0; i < rows; i++) {
        double sum = 0.0; 
        for (long j = 0; j < cols; j++) {
            sum += (double)mat[i * cols + j] * (double)vec[j];
        }
        out[i] = (float)sum;
    }
}

static float *alloc_f32(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 32, n * sizeof(float)) != 0) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    return (float *)p;
}

static void fill_random(float *a, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        a[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
    }
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int run_correctness(long rows, long cols, unsigned seed) {
    float *mat = alloc_f32((size_t)rows * cols);
    float *vec = alloc_f32(cols);
    float *out_ref = alloc_f32(rows);
    float *out_scalar = alloc_f32(rows);
    float *out_simd = alloc_f32(rows);

    fill_random(mat, (size_t)rows * cols, seed);
    fill_random(vec, cols, seed + 1);

    matvec_ref(mat, vec, out_ref, rows, cols);
    matvec_scalar(mat, vec, out_scalar, rows, cols);
    matvec_simd(mat, vec, out_simd, rows, cols);

    double err_scalar = max_abs_diff(out_ref, out_scalar, rows);
    double err_simd = max_abs_diff(out_ref, out_simd, rows);

    const double tol = 1e-2;
    int ok = (err_scalar < tol) && (err_simd < tol);

    printf("  rows=%-5ld cols=%-5ld  max_err(scalar)=%.6f  max_err(simd)=%.6f  %s\n",
           rows, cols, err_scalar, err_simd, ok ? "PASS" : "FAIL");

    free(mat); free(vec); free(out_ref); free(out_scalar); free(out_simd);
    return ok;
}

typedef struct {
    long rows, cols;
    double scalar_ns, simd_ns;
} bench_row_t;

static bench_row_t run_benchmark(long rows, long cols, int iters, unsigned seed) {
    float *mat = alloc_f32((size_t)rows * cols);
    float *vec = alloc_f32(cols);
    float *out = alloc_f32(rows);

    fill_random(mat, (size_t)rows * cols, seed);
    fill_random(vec, cols, seed + 1);

    // warm up (populate caches, page faults) before timing
    matvec_scalar(mat, vec, out, rows, cols);
    matvec_simd(mat, vec, out, rows, cols);

    double t0 = now_seconds();
    for (int k = 0; k < iters; k++) matvec_scalar(mat, vec, out, rows, cols);
    double t1 = now_seconds();
    for (int k = 0; k < iters; k++) matvec_simd(mat, vec, out, rows, cols);
    double t2 = now_seconds();

    bench_row_t r;
    r.rows = rows;
    r.cols = cols;
    r.scalar_ns = (t1 - t0) / iters * 1e9;
    r.simd_ns = (t2 - t1) / iters * 1e9;

    free(mat); free(vec); free(out);
    return r;
}

int main(void) {
    printf("=== Correctness check (vs. double-accumulated C reference) ===\n");
    int all_ok = 1;
    all_ok &= run_correctness(4, 8, 1);
    all_ok &= run_correctness(16, 37, 2);     // cols not a multiple of 8
    all_ok &= run_correctness(128, 1029, 3);  // exercises tail loop at scale
    all_ok &= run_correctness(1, 1, 4);       // degenerate 1x1 case
    if (!all_ok) {
        fprintf(stderr, "\nCorrectness check FAILED, aborting benchmark.\n");
        return 1;
    }
    printf("All correctness checks passed.\n\n");

    printf("=== Benchmark: matvec_scalar vs matvec_simd (AVX2 + FMA) ===\n");
    printf("%-10s %-10s %14s %14s %10s %14s %14s\n",
           "rows", "cols", "scalar(ns)", "simd(ns)", "speedup",
           "scalar_GF/s", "simd_GF/s");

    long sizes[][2] = {
        {64, 64}, {128, 128}, {256, 256}, {512, 512},
        {1024, 1024}, {2048, 2048}, {4096, 4096},
        {4096, 1000},   // non-multiple-of-8 column count under real load
    };
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    FILE *csv = fopen("results/benchmark_results.csv", "w");
    fprintf(csv, "rows,cols,scalar_ns,simd_ns,speedup,scalar_gflops,simd_gflops\n");

    for (int i = 0; i < n_sizes; i++) {
        long rows = sizes[i][0], cols = sizes[i][1];
        long total_elems = rows * cols;
        int iters = total_elems < 200000 ? 2000 : (total_elems < 2000000 ? 200 : 30);

        bench_row_t r = run_benchmark(rows, cols, iters, 100 + i);
        double flops = 2.0 * (double)rows * (double)cols; // 1 mul + 1 add per element
        double scalar_gflops = flops / r.scalar_ns; // ns -> GFLOP/s cancels 1e9/1e9
        double simd_gflops = flops / r.simd_ns;
        double speedup = r.scalar_ns / r.simd_ns;

        printf("%-10ld %-10ld %14.1f %14.1f %9.2fx %14.2f %14.2f\n",
               rows, cols, r.scalar_ns, r.simd_ns, speedup,
               scalar_gflops, simd_gflops);

        fprintf(csv, "%ld,%ld,%.2f,%.2f,%.4f,%.4f,%.4f\n",
                rows, cols, r.scalar_ns, r.simd_ns, speedup,
                scalar_gflops, simd_gflops);
    }
    fclose(csv);

    printf("\nRaw results written to results/benchmark_results.csv\n");
    return 0;
}
