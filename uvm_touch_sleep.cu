// uvm_touch_sleep.cu
//
// Allocates a buffer in Unified Virtual Memory (UVM) via cudaMallocManaged,
// touches every page of it from a GPU kernel (forcing each page to be
// faulted in / migrated to device memory), then sleeps for 60 seconds so
// you can inspect resident state with nvidia-smi, nuvmtop, cgroup memory
// counters, etc.
//
// Build:
//   nvcc -O2 -arch=sm_native uvm_touch_sleep.cu -o uvm_touch_sleep
//   (or set -arch=sm_XX explicitly, e.g. sm_86 / sm_120)
//
// Run:
//   ./uvm_touch_sleep [size_in_MB]
//
// Default size is 1024 MB if no argument is given.

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
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

static const size_t PAGE_SIZE = 4096; // bytes, matches typical UVM page size

// Grid-stride kernel: each thread walks a subset of pages and writes one
// byte per page. Writing (not just reading) guarantees the page is
// faulted in as read-write and actually migrated/resident on the device,
// not just mapped read-only.
__global__ void touch_pages_kernel(char *buf, size_t num_pages, size_t page_size) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t page = tid; page < num_pages; page += stride) {
        size_t offset = page * page_size;
        buf[offset] = (char)(page & 0xFF); // touch first byte of the page
    }
}

int main(int argc, char **argv) {
    size_t size_mb = 1024; // default 1 GiB
    if (argc > 1) {
        size_mb = (size_t)strtoull(argv[1], nullptr, 10);
        if (size_mb == 0) {
            fprintf(stderr, "Invalid size_in_MB argument, using default 1024 MB\n");
            size_mb = 1024;
        }
    }

    size_t total_bytes = size_mb * 1024ULL * 1024ULL;
    size_t num_pages = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

    printf("PID: %d\n", getpid());
    printf("Allocating %zu MB (%zu bytes) in UVM, %zu pages of %zu bytes\n",
           size_mb, total_bytes, num_pages, PAGE_SIZE);

    char *buf = nullptr;
    CUDA_CHECK(cudaMallocManaged(&buf, total_bytes));

    // Kernel launch config: enough threads to cover pages with a grid-stride loop.
    int threads_per_block = 256;
    int blocks = 1024; // grid-stride loop covers the rest regardless of page count
    if ((size_t)(blocks * threads_per_block) > num_pages) {
        blocks = (int)((num_pages + threads_per_block - 1) / threads_per_block);
        if (blocks < 1) blocks = 1;
    }

    printf("Launching kernel to touch all pages (%d blocks x %d threads)...\n",
           blocks, threads_per_block);

    touch_pages_kernel<<<blocks, threads_per_block>>>(buf, num_pages, PAGE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("All %zu pages touched. Sleeping for 60 seconds...\n", num_pages);
    fflush(stdout);
    sleep(60);

    printf("Done sleeping. Freeing memory and exiting.\n");
    CUDA_CHECK(cudaFree(buf));

    return 0;
}
