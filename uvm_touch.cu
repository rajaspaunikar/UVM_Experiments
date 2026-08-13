// uvm_touch.cu
//
// Allocates a UVM (managed) memory region of a size given on the command
// line, touches every page of it from a GPU kernel (forcing physical page
// population/migration), then sleeps for 60 seconds before exiting.
//
// Usage:
//   ./uvm_touch <size>
//
// <size> accepts a plain byte count, or a suffix of K/M/G (base-1024),
// e.g. "512M", "2G", "1073741824".
//
// Build:
//   nvcc -O2 -arch=native uvm_touch.cu -o uvm_touch
//   (or set -arch=sm_XX for your GPU, e.g. sm_120 for RTX 5060 Ti)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <unistd.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Parse a size string like "512M", "2G", "1024K", or a plain byte count.
static size_t parse_size(const char *s) {
    char *end = nullptr;
    double val = strtod(s, &end);
    if (end == s) {
        fprintf(stderr, "Invalid size argument: %s\n", s);
        exit(EXIT_FAILURE);
    }

    size_t multiplier = 1;
    if (*end != '\0') {
        switch (*end) {
            case 'k': case 'K': multiplier = 1ULL << 10; break;
            case 'm': case 'M': multiplier = 1ULL << 20; break;
            case 'g': case 'G': multiplier = 1ULL << 30; break;
            case 't': case 'T': multiplier = 1ULL << 40; break;
            case 'b': case 'B': multiplier = 1; break;
            default:
                fprintf(stderr, "Unknown size suffix: %s\n", end);
                exit(EXIT_FAILURE);
        }
    }

    return static_cast<size_t>(val * multiplier);
}

// Touch every page of the buffer: write one value per page (using the
// system page size, 4KiB, which is <= the GPU page size, so every GPU
// page ends up touched too).
__global__ void touch_pages_kernel(uint8_t *buf, size_t size, size_t stride) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t offset = idx * stride;
    if (offset < size) {
        buf[offset] = static_cast<uint8_t>(idx & 0xFF);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size>  (e.g. 512M, 2G, 1073741824)\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t size = parse_size(argv[1]);
    if (size == 0) {
        fprintf(stderr, "Size must be greater than 0\n");
        return EXIT_FAILURE;
    }

    const size_t PAGE_SIZE = 4096; // touch granularity
    size_t num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;

    printf("Requested size: %zu bytes (%.2f MiB)\n", size, size / (1024.0 * 1024.0));
    printf("Touching %zu pages (stride = %zu bytes)\n", num_pages, PAGE_SIZE);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU %d: %s\n", device, prop.name);

    uint8_t *buf = nullptr;
    CUDA_CHECK(cudaMallocManaged(&buf, size));
    printf("cudaMallocManaged succeeded, ptr = %p\n", (void *)buf);

    // Optional hint: this memory will be accessed by the GPU device.
    // CUDA_CHECK(cudaMemAdvise(buf, size, cudaMemAdviseSetPreferredLocation, device));

    int threads = 256;
    int blocks = static_cast<int>((num_pages + threads - 1) / threads);

    printf("Launching touch kernel: %d blocks x %d threads\n", blocks, threads);
    touch_pages_kernel<<<blocks, threads>>>(buf, size, PAGE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("All pages touched.\n");

    printf("Sleeping for 60 seconds (memory stays allocated and resident)...\n");
    fflush(stdout);
    sleep(60);

    CUDA_CHECK(cudaFree(buf));
    printf("Freed managed memory. Exiting.\n");

    return EXIT_SUCCESS;
}
