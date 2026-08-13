#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(call)                                                     \
do {                                                                     \
cudaError_t _e = (call);                                             \
if (_e != cudaSuccess) {                                             \
printf("%s failed: %s\n", #call, cudaGetErrorString(_e));        \
return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void touch_kernel(char* p, size_t n) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        p[i] = 1;
    }
}

// ---- stats helpers ----
struct Stats {
    double mean;
    double stddev;
    double p50;
    double p95;
    double p99;
    double min;
    double max;
};

Stats compute_stats(std::vector<float>& samples) {
    Stats s{};
    size_t n = samples.size();
    if (n == 0) return s;

    std::vector<float> sorted = samples; // copy for percentile calc
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;
    for (float v : samples) sum += v;
    s.mean = sum / n;

    double sq_sum = 0.0;
    for (float v : samples) sq_sum += (v - s.mean) * (v - s.mean);
    s.stddev = std::sqrt(sq_sum / n);   // population stddev; use /(n-1) for sample stddev

    auto percentile = [&](double p) -> double {
        if (n == 1) return sorted[0];
        double idx = p * (n - 1); // linear interpolation method
        size_t lo = (size_t)std::floor(idx);
        size_t hi = (size_t)std::ceil(idx);
        if (lo == hi) return sorted[lo];
        double frac = idx - lo;
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
    };

    s.p50 = percentile(0.50);
    s.p95 = percentile(0.95);
    s.p99 = percentile(0.99);
    s.min = sorted.front();
    s.max = sorted.back();

    return s;
}

int main(int argc, char** argv) {
    size_t alloc_mb    = (argc > 1) ? atoll(argv[1]) : 1024;
    size_t iterations  = (argc > 2) ? atoll(argv[2]) : 1;
    size_t alloc_bytes = alloc_mb * 1024 * 1024;

    auto t_wall_start = std::chrono::high_resolution_clock::now();

    void* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, alloc_bytes));
    printf("Allocated %zu MB of managed memory at %p\n", alloc_mb, ptr);

    size_t threads = 256;
    size_t blocks  = 1024;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::vector<float> kernel_times_ms;
    kernel_times_ms.reserve(iterations);

    for (size_t it = 0; it < iterations; it++) {
        cudaEventRecord(ev_start);
        touch_kernel<<<blocks, threads>>>((char*)ptr, alloc_bytes);
        cudaEventRecord(ev_stop);

        CUDA_CHECK(cudaGetLastError());
        cudaEventSynchronize(ev_stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        kernel_times_ms.push_back(ms);

        printf("  iter %zu: %.3f ms (%.2f GB/s)\n",
               it, ms, (alloc_bytes / 1e9) / (ms / 1000.0));
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    Stats s = compute_stats(kernel_times_ms);
    printf("\n--- Kernel timing stats over %zu iterations ---\n", iterations);
    printf("mean   : %.3f ms\n", s.mean);
    printf("stddev : %.3f ms\n", s.stddev);
    printf("min    : %.3f ms\n", s.min);
    printf("p50    : %.3f ms\n", s.p50);
    printf("p95    : %.3f ms\n", s.p95);
    printf("p99    : %.3f ms\n", s.p99);
    printf("max    : %.3f ms\n", s.max);

    sleep(60);

    CUDA_CHECK(cudaFree(ptr));

    auto t_wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();
    printf("\nTotal process wall-clock time (alloc+touch+sleep+free): %.3f ms\n", wall_ms);

    return 0;
}