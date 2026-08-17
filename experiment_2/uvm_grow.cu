// uvm_grow.cu
//
// Simulates a process that starts small and gradually increases its UVM
// memory footprint over time while doing real work on the memory it holds
// — useful for testing BoxD's enforcement as usage crosses soft/hard limits
// live, rather than starting already oversubscribed.
//
// Strategy: allocate a chunk, touch it (do "work"), sleep briefly, allocate
// another chunk, touch everything again (old + new), repeat until max_MB is
// reached or duration expires. Then hold steady at the final size.
//
// Build:
//   nvcc -O2 -arch=sm_native uvm_grow.cu -o uvm_grow
//
// Run:
//   ./uvm_grow [chunk_MB] [growth_interval_sec] [max_MB] [hold_sec]
//
// Example - grow by 500MB every 3s up to 6000MB, then hold for 60s:
//   ./uvm_grow 500 3 6000 60
//
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

// Touches every page of a buffer, simulating "doing work" on it (e.g. a
// forward pass touching activations, or a service touching cached state).
__global__ void touch_kernel(char *buf, size_t num_pages, size_t page_size,
                              uint64_t seed) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t page = tid; page < num_pages; page += stride) {
        size_t offset = page * page_size;
        buf[offset] = (char)((page ^ seed) & 0xFF);
    }
}

struct Chunk {
    char *ptr;
    size_t bytes;
};

static void touch_all(std::vector<Chunk> &chunks, uint64_t seed) {
    int threads_per_block = 256;
    int blocks = 512;
    for (auto &c : chunks) {
        size_t num_pages = (c.bytes + PAGE_SIZE - 1) / PAGE_SIZE;
        touch_kernel<<<blocks, threads_per_block>>>(c.ptr, num_pages, PAGE_SIZE, seed);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
    size_t chunk_mb          = argc > 1 ? strtoull(argv[1], nullptr, 10) : 500;
    int growth_interval_sec  = argc > 2 ? atoi(argv[2]) : 3;
    size_t max_mb            = argc > 3 ? strtoull(argv[3], nullptr, 10) : 8000;
    int hold_sec             = argc > 4 ? atoi(argv[4]) : 60;

    size_t chunk_bytes = chunk_mb * 1024ULL * 1024ULL;
    size_t max_bytes   = max_mb   * 1024ULL * 1024ULL;

    printf("PID: %d\n", getpid());
    printf("Growing by %zu MB every %ds, up to %zu MB, then holding %ds\n",
           chunk_mb, growth_interval_sec, max_mb, hold_sec);

    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    printf("GPU memory: %.1f MB free / %.1f MB total\n", free_b / 1e6, total_b / 1e6);

    std::vector<Chunk> chunks;
    size_t current_bytes = 0;
    uint64_t step = 0;

    printf("Starting growth loop...\n");
    fflush(stdout);

    while (current_bytes < max_bytes) {
        char *new_ptr = nullptr;
        CUDA_CHECK(cudaMallocManaged(&new_ptr, chunk_bytes));
        chunks.push_back({new_ptr, chunk_bytes});
        current_bytes += chunk_bytes;
        step++;

        // Touch everything allocated so far — old chunks stay "hot"/resident,
        // new chunk gets faulted in for the first time.
        touch_all(chunks, step);

        printf("  step=%llu total_footprint=%zu MB (%zu chunks)\n",
               (unsigned long long)step, current_bytes / (1024 * 1024), chunks.size());
        fflush(stdout);

        sleep(growth_interval_sec);
    }

    printf("Reached target footprint (%zu MB). Holding steady for %ds, "
           "touching all chunks periodically...\n",
           current_bytes / (1024 * 1024), hold_sec);
    fflush(stdout);

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    do {
        touch_all(chunks, ++step);
        sleep(1);
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while ((now.tv_sec - start.tv_sec) < hold_sec);

    printf("Done. Final footprint: %zu MB across %zu chunks. Freeing and exiting.\n",
           current_bytes / (1024 * 1024), chunks.size());

    for (auto &c : chunks)
        CUDA_CHECK(cudaFree(c.ptr));

    return 0;
}