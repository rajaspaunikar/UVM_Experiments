/*
 * uvm_page_touch.cu
 *
 * Allocates a given amount of CUDA Unified (Managed) Memory and traverses
 * every page in it a given number of times (passes), touching one word
 * per page from the GPU. Useful for generating controlled UVM fault /
 * eviction / migration traffic to observe with nuvmtop, cgroup GPU memory
 * accounting (BoxD), or nvidia-smi.
 *
 * Build:
 *   nvcc -O2 -arch=native uvm_page_touch.cu -o uvm_page_touch
 *   (or set -arch=sm_XX for your GPU, e.g. sm_120 for Blackwell, sm_86 for A4000)
 *
 * Usage:
 *   ./uvm_page_touch <size_MB> <passes> [page_size_bytes] [--prefetch]
 *
 * Examples:
 *   ./uvm_page_touch 1024 1            # touch 1GB, one pass
 *   ./uvm_page_touch 2048 2            # touch 2GB, two passes (re-touch each page)
 *   ./uvm_page_touch 512 2 4096 --prefetch  # prefetch to GPU before each pass
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// One thread touches one page: writes then reads a single 8-byte word
// at the start of that page. This is enough to force the UVM driver to
// resolve a page fault and migrate/populate the page.
__global__ void touch_pages_kernel(char* base, size_t num_pages, size_t page_size,
                                    unsigned long long pass_id) {
    size_t page_idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (page_idx >= num_pages) return;

    unsigned long long* word =
        reinterpret_cast<unsigned long long*>(base + page_idx * page_size);

    // Write something that depends on the pass so the compiler can't
    // optimize this away, and so you can sanity-check contents later.
    *word = pass_id * 1000003ULL + page_idx;

    // Read it back to also generate a read access on the same page.
    volatile unsigned long long readback = *word;
    (void)readback;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <size_MB> <passes> [page_size_bytes] [--prefetch]\n"
            "  size_MB          total managed memory to allocate, in MiB\n"
            "  passes            number of times to traverse all pages (e.g. 1 or 2)\n"
            "  page_size_bytes   optional, default 4096 (must match/divide your UVM page size)\n"
            "  --prefetch        optional, cudaMemPrefetchAsync the whole region to the GPU\n"
            "                    before each pass (forces bulk migration instead of\n"
            "                    per-page fault-driven migration)\n",
            prog);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    size_t size_mb = strtoull(argv[1], nullptr, 10);
    int passes = atoi(argv[2]);
    size_t page_size = 4096;
    bool do_prefetch = false;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--prefetch") == 0) {
            do_prefetch = true;
        } else {
            page_size = strtoull(argv[i], nullptr, 10);
        }
    }

    if (size_mb == 0 || passes <= 0 || page_size == 0) {
        fprintf(stderr, "Invalid arguments.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    size_t total_bytes = size_mb * 1024ULL * 1024ULL;
    size_t num_pages = (total_bytes + page_size - 1) / page_size;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Requested size: %zu MB (%zu bytes)\n", size_mb, total_bytes);
    printf("Page size: %zu bytes, num pages: %zu\n", page_size, num_pages);
    printf("Passes: %d, prefetch before each pass: %s\n", passes,
           do_prefetch ? "yes" : "no");

    char* d_ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_ptr, total_bytes));

    // CUDA 12.2+/13.x: cudaMemAdvise / cudaMemPrefetchAsync take a cudaMemLocation
    // struct instead of a bare device int.
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    // Optional: advise the driver this memory will be accessed mostly by the GPU.
    // Comment out if you want default (first-touch) migration behavior.
    CUDA_CHECK(cudaMemAdvise(d_ptr, total_bytes, cudaMemAdviseSetPreferredLocation, loc));

    const int threads_per_block = 256;
    const size_t blocks =
        (num_pages + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int p = 0; p < passes; ++p) {
        if (do_prefetch) {
            CUDA_CHECK(cudaMemPrefetchAsync(d_ptr, total_bytes, loc, 0, 0));
        }

        CUDA_CHECK(cudaEventRecord(start));
        touch_pages_kernel<<<blocks, threads_per_block>>>(
            d_ptr, num_pages, page_size, static_cast<unsigned long long>(p));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        double gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
        double gbps = (ms > 0.0f) ? (gb / (ms / 1000.0)) : 0.0;

        printf("Pass %d/%d: %zu pages touched in %.3f ms (%.2f GB/s effective touch rate)\n",
               p + 1, passes, num_pages, ms, gbps);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_ptr));

    printf("Done.\n");
    return EXIT_SUCCESS;
}
