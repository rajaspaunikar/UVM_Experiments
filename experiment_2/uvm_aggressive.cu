#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <getopt.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

/**
 * XORShift PRNG for GPU threads.
 * Generates non-deterministic, scattered accesses across the allocation
 * to defeat hardware prefetch heuristics and locality tracking.
 */
__device__ inline uint64_t xorshift64(uint64_t state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

/**
 * Aggressive UVM Access Kernel:
 * - Scatters writes unpredictable across the entire current working set.
 * - Forces Read-Write migration faults on every page access.
 */
__global__ void uvm_scatter_write_kernel(uint64_t *data, size_t num_elements, uint64_t seed_offset) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    // Unique per-thread seed combined with iteration offset
    uint64_t rng_state = tid + seed_offset + 0x9E3779B97F4A7C15ULL;
    
    // Perform multiple unaligned, scattered writes per thread
    for (int i = 0; i < 8; ++i) {
        rng_state = xorshift64(rng_state);
        size_t target_idx = rng_state % num_elements;

        // Force a Read-Write fault & migration (Trait #4: Write Faults)
        data[target_idx] += static_cast<uint64_t>(tid + i);
    }
}

/**
 * Optional CPU Thrash Kernel:
 * Touches random pages from the Host CPU while the GPU runs, forcing cross-node
 * host-vs-device ping-pong thrashing over UVM.
 */
void cpu_thrash_sidechannel(uint64_t *data, size_t num_elements, bool enabled) {
    if (!enabled) return;
    
    // CPU touches scattered pages in the managed buffer
    for (size_t i = 0; i < 1024; ++i) {
        size_t idx = (rand() * 4096) % num_elements;
        data[idx] += 1; // Forces page fault back to Host System Memory
    }
}

int main(int argc, char *argv[]) {
    // Default Configuration Parameters
    size_t initial_gb = 12;        // Initial UVM allocation in GB
    size_t max_gb = 12;           // Maximum expansion limit in GB (Oversubscription)
    int duration_sec = 60;        // Sustained test duration
    bool enable_growth = true;    // Trait #5: Dynamic footprint expansion
    bool enable_cpu_pingpong = false; // Trait #6: Host vs GPU contention

    std::cout << "===================================================\n";
    std::cout << "      UVM Aggressive Memory Stress Tester          \n";
    std::cout << "===================================================\n";

    // Detect GPU physical memory for context
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "[+] Target GPU: " << prop.name << "\n";
    std::cout << "[+] Total VRAM: " << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << " GB\n";

    // Compute element sizes (uint64_t = 8 bytes)
    size_t current_gb = initial_gb;
    size_t num_elements = (current_gb * 1024 * 1024 * 1024) / sizeof(uint64_t);

    uint64_t *uvm_buffer = nullptr;
    
    // Trait #1: Managed Memory Allocation (Oversubscribed / cgroup-targeted)
    std::cout << "[+] Allocating initial " << current_gb << " GB Managed Memory (UVM)...\n";
    CUDA_CHECK(cudaMallocManaged(&uvm_buffer, num_elements * sizeof(uint64_t)));

    // Ensure memory is un-prefetched to force cold faults initially
    CUDA_CHECK(cudaMemset(uvm_buffer, 0, num_elements * sizeof(uint64_t)));

    auto start_time = std::chrono::steady_clock::now();
    uint64_t iteration = 0;

    std::cout << "[+] Beginning sustained aggressive fault generation for " 
              << duration_sec << " seconds...\n\n";

    int threads_per_block = 256;

    while (true) {
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(current_time - start_time).count();
        if (elapsed_sec >= duration_sec) break;

        // Trait #5: Grow footprint over time to outpace memory budget
        if (enable_growth && elapsed_sec > (duration_sec / 2) && current_gb < max_gb) {
            size_t new_gb = max_gb;
            size_t new_num_elements = (new_gb * 1024 * 1024 * 1024) / sizeof(uint64_t);

            std::cout << "\n[!] [Trait 5] EXPANDING FOOTPRINT: " << current_gb 
                      << " GB -> " << new_gb << " GB (Triggering BoxD/Driver Evictions)...\n";

            uint64_t *new_buffer = nullptr;
            CUDA_CHECK(cudaMallocManaged(&new_buffer, new_num_elements * sizeof(uint64_t)));
            
            // Free old, assign new
            CUDA_CHECK(cudaFree(uvm_buffer));
            uvm_buffer = new_buffer;
            num_elements = new_num_elements;
            current_gb = new_gb;
        }

        int blocks = (num_elements / 1024 + threads_per_block - 1) / threads_per_block;
        if (blocks > 65535) blocks = 65535; // Cap launch dimensions

        // Trait #2 & #4: Launch kernel with random writes
        uvm_scatter_write_kernel<<<blocks, threads_per_block>>>(
            uvm_buffer, num_elements, iteration * 1337
        );

        // Trait #6: Concurrent CPU side-channel writes causing Host<->GPU migrations
        cpu_thrash_sidechannel(uvm_buffer, num_elements, enable_cpu_pingpong);

        // Trait #3: Sustained looping (Synchronize per iteration to force immediate fault processing)
        CUDA_CHECK(cudaDeviceSynchronize());

        iteration++;
        if (iteration % 50 == 0) {
            std::cout << "  -> Progress: " << elapsed_sec << "s / " << duration_sec 
                      << "s | Iterations: " << iteration 
                      << " | Current Size: " << current_gb << " GB\r" << std::flush;
        }
    }

    std::cout << "\n\n[+] Test complete. Executed " << iteration << " high-pressure fault cycles.\n";
    std::cout << "[+] Cleaning up UVM allocations...\n";
    CUDA_CHECK(cudaFree(uvm_buffer));

    return 0;
}