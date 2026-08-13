// uvm_aggressive.cu
//
// Simulates a process with aggressive/thrashing UVM memory behavior, for
// stress-testing BoxD's cgroup-based GPU memory accounting and enforcement.
//
// Strategy: allocate a UVM buffer (optionally larger than GPU memory or your
// cgroup's hard limit, to force eviction), then repeatedly launch a touch
// kernel over it in a loop for a configurable duration, using a configurable
// access pattern (sequential / random / strided) to control fault behavior.
//
// Build:
//   nvcc -O2 -arch=sm_native uvm_aggressive.cu -o uvm_aggressive
//
// Run:
//   ./uvm_aggressive [size_MB] [duration_sec] [pattern] [stride_pages]
//   pattern: 0=sequential 1=random 2=strided (default 0)
//
// Example - force oversubscription + random thrash for 120s:
//   ./uvm_aggressive 20000 120 1
//
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

static const size_t PAGE_SIZE = 4096;

enum Pattern { SEQUENTIAL = 0, RANDOM = 1, STRIDED = 2 };

// Simple xorshift PRNG usable on the device, seeded per-thread.
__device__ __forceinline__ uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

__global__ void touch_kernel(char *buf, size_t num_pages, size_t page_size,
                              int pattern, size_t stride_pages, uint64_t seed,
                              unsigned long long *fault_counter) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total_threads = (size_t)gridDim.x * blockDim.x;
    uint64_t rng_state = seed ^ (tid * 0x9E3779B97F4A7C15ULL);

    // Each thread touches a fixed number of pages per kernel launch,
    // chosen according to the access pattern.
    const size_t touches_per_thread = 64;

    for (size_t i = 0; i < touches_per_thread; ++i) {
        size_t page;
        switch (pattern) {
            case RANDOM:
                page = xorshift64(&rng_state) % num_pages;
                break;
            case STRIDED:
                page = (tid * stride_pages + i * total_threads) % num_pages;
                break;
            case SEQUENTIAL:
            default:
                page = (tid + i * total_threads) % num_pages;
                break;
        }
        size_t offset = page * page_size;
        // Write, not read: forces RW fault + migration, not just a mapping.
        buf[offset] = (char)((page + i) & 0xFF);
        atomicAdd(fault_counter, 1ULL);
    }
}

int main(int argc, char **argv) {
    size_t size_mb        = argc > 1 ? strtoull(argv[1], nullptr, 10) : 4096;
    int duration_sec      = argc > 2 ? atoi(argv[2]) : 60;
    int pattern            = argc > 3 ? atoi(argv[3]) : SEQUENTIAL;
    size_t stride_pages   = argc > 4 ? strtoull(argv[4], nullptr, 10) : 512;

    size_t total_bytes = size_mb * 1024ULL * 1024ULL;
    size_t num_pages    = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

    printf("PID: %d\n", getpid());
    printf("Allocating %zu MB (%zu bytes), %zu pages, pattern=%d, duration=%ds\n",
           size_mb, total_bytes, num_pages, pattern, duration_sec);

    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    printf("GPU memory: %.1f MB free / %.1f MB total%s\n",
           free_b / 1e6, total_b / 1e6,
           total_bytes > total_b ? "  [OVERSUBSCRIBING]" : "");

    char *buf = nullptr;
    CUDA_CHECK(cudaMallocManaged(&buf, total_bytes));

    // Optional: hint the driver this buffer prefers GPU residency, to make
    // eviction/thrash pressure show up faster under oversubscription.
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = device;

    cudaMemAdvise(buf, total_bytes, cudaMemAdviseSetPreferredLocation, loc);

    unsigned long long *d_fault_counter;
    CUDA_CHECK(cudaMalloc(&d_fault_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_fault_counter, 0, sizeof(unsigned long long)));

    int threads_per_block = 256;
    int blocks = 512;

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    uint64_t launch_count = 0;

    printf("Starting thrash loop...\n");
    fflush(stdout);

    do {
        touch_kernel<<<blocks, threads_per_block>>>(
            buf, num_pages, PAGE_SIZE, pattern, stride_pages,
            (uint64_t)launch_count * 0x2545F4914F6CDD1DULL, d_fault_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        launch_count++;

        if (launch_count % 50 == 0) {
            unsigned long long faults;
            CUDA_CHECK(cudaMemcpy(&faults, d_fault_counter, sizeof(faults),
                                   cudaMemcpyDeviceToHost));
            printf("  launches=%llu total_touches=%llu\n",
                   (unsigned long long)launch_count, faults);
            fflush(stdout);
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
    } while ((now.tv_sec - start.tv_sec) < duration_sec);

    unsigned long long final_faults;
    CUDA_CHECK(cudaMemcpy(&final_faults, d_fault_counter, sizeof(final_faults),
                           cudaMemcpyDeviceToHost));
    printf("Done. %llu kernel launches, %llu total page touches over %ds\n",
           (unsigned long long)launch_count, final_faults, duration_sec);

    CUDA_CHECK(cudaFree(d_fault_counter));
    CUDA_CHECK(cudaFree(buf));
    return 0;
}