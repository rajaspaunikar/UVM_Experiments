/*
 * p1_touch.cu — P1: allocate 12GB managed memory, touch all pages 1 time,
 * pacing page-touches in small chunks with a short sleep between each chunk
 * so total wall time is ~25 seconds.
 *
 * Build: nvcc -O2 -arch=native p1_touch.cu -o p1_touch
 * Run:   ./p1_touch
 *        (or under a cgroup: ./run_in_cgroup.sh <cgroup_name> ./p1_touch)
 */

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static const size_t SIZE_MB          = 8ULL * 1024ULL; // 12 GB
static const int    ITERATIONS       = 1;
static const size_t PAGE_SIZE        = 4096;
static const double TARGET_SECONDS   = 25.0;  // total wall-clock target for this process
static const int    CHUNKS_PER_PASS  = 200;   // how many touch+sleep steps per pass
static const char*  TAG              = "[P1]";

// Touches pages [page_offset, page_offset + chunk_count) only.
__global__ void touch_pages_chunk_kernel(char* base, size_t page_offset,
                                          size_t chunk_count, size_t page_size,
                                          unsigned long long pass_id) {
    size_t local_idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (local_idx >= chunk_count) return;
    size_t page_idx = page_offset + local_idx;

    unsigned long long* word =
        reinterpret_cast<unsigned long long*>(base + page_idx * page_size);
    *word = pass_id * 1000003ULL + page_idx;
    volatile unsigned long long readback = *word;
    (void)readback;
}

int main() {
    size_t total_bytes = SIZE_MB * 1024ULL * 1024ULL;
    size_t num_pages = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("%s pid=%d device=%s size=%zuMB pages=%zu iterations=%d target=%.1fs\n",
           TAG, getpid(), prop.name, SIZE_MB, num_pages, ITERATIONS, TARGET_SECONDS);

    char* d_ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_ptr, total_bytes));

    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;
    CUDA_CHECK(cudaMemAdvise(d_ptr, total_bytes, cudaMemAdviseSetPreferredLocation, loc));

    const int threads_per_block = 256;

    // Pages per chunk, and the sleep interval between chunks, so that
    // CHUNKS_PER_PASS chunks * (kernel time + sleep) ~= TARGET_SECONDS/ITERATIONS
    // per pass.
    size_t chunk_pages = (num_pages + CHUNKS_PER_PASS - 1) / CHUNKS_PER_PASS;
    double target_seconds_per_pass = TARGET_SECONDS / ITERATIONS;
    unsigned long long chunk_sleep_us = static_cast<unsigned long long>(
        (target_seconds_per_pass / CHUNKS_PER_PASS) * 1e6);

    printf("%s chunk_pages=%zu chunk_sleep=%.2f ms (%d chunks/pass)\n",
           TAG, chunk_pages, chunk_sleep_us / 1000.0, CHUNKS_PER_PASS);

    auto wall_start = std::chrono::steady_clock::now();

    for (int p = 0; p < ITERATIONS; ++p) {
        size_t pages_done = 0;
        int chunk_idx = 0;
        double pass_kernel_ms = 0.0;

        while (pages_done < num_pages) {
            size_t this_chunk = std::min(chunk_pages, num_pages - pages_done);
            size_t blocks = (this_chunk + threads_per_block - 1) / threads_per_block;

            cudaEvent_t cstart, cstop;
            CUDA_CHECK(cudaEventCreate(&cstart));
            CUDA_CHECK(cudaEventCreate(&cstop));

            CUDA_CHECK(cudaEventRecord(cstart));
            touch_pages_chunk_kernel<<<blocks, threads_per_block>>>(
                d_ptr, pages_done, this_chunk, PAGE_SIZE,
                static_cast<unsigned long long>(p));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(cstop));
            CUDA_CHECK(cudaEventSynchronize(cstop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, cstart, cstop));
            pass_kernel_ms += ms;
            CUDA_CHECK(cudaEventDestroy(cstart));
            CUDA_CHECK(cudaEventDestroy(cstop));

            usleep(chunk_sleep_us);

            pages_done += this_chunk;
            chunk_idx++;
        }

        printf("%s pass %d/%d: %zu pages touched across %d chunks, kernel time %.2f ms\n",
               TAG, p + 1, ITERATIONS, num_pages, chunk_idx, pass_kernel_ms);
    }

    auto wall_end = std::chrono::steady_clock::now();
    double total_wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    CUDA_CHECK(cudaFree(d_ptr));

    printf("%s done. total wall time: %.2f s\n", TAG, total_wall_s);
    return EXIT_SUCCESS;
}
