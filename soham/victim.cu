#include <iostream>
#include <cuda.h>
using namespace std;

__global__ void accessAll(int *data, long long n)
{
    // A single thread in a single block accesses the entire array sequentially
    for (long long i = 0; i < n; i++)
    {
        data[i] += 1;
    }
}

int main()
{
    int current_device = 0;
    cudaSetDevice(current_device); // Or cudaInitDevice

    // STEP 1: Start with an initial set of resources, e.g., by fetching the available resources of the GPU
    cudaDevResource initial_SM_resources = {};
    cudaDeviceGetDevResource(current_device /* GPU device */,&initial_SM_resources /* device resource to populate */,cudaDevResourceTypeSm /* resource type - part of struct cudaDevResource*/);

    std::cout << "Initial SM resources: " << initial_SM_resources.sm.smCount << " SMs" << std::endl; // number of available SMs

    // Special fields relevant for partitioning (see Step 3 below)
    // std::cout << "Min. SM partition size: " <<  initial_SM_resources.sm.minSmPartitionSize << " SMs" << std::endl;
    // std::cout << "SM co-scheduled alignment: " <<  initial_SM_resources.sm.smCoscheduledAlignment << " SMs" << std::endl;

    // STEP 2: Partition the SM resources into one or more partitions (using one of the available split APIs).

    cudaDevResource actual_split_result[2] = {{}, {}};
    cudaDevResource remaining_partition = {};

    // M1
    unsigned int min_SM_count = 8;
    unsigned int actual_split_groups = 2; // groups can be reduced, try with 16 groups 1 SM each
    cudaDevSmResourceSplitByCount(&actual_split_result[0],&actual_split_groups,&initial_SM_resources,&remaining_partition,0 /*useFlags */,min_SM_count);

    //M2
    // cudaDevSmResourceGroupParams group_params_use_case[2] = {{.smCount = 14, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0},
                                                        //  {.smCount = 2, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0}};
    // cudaDevSmResourceSplit(&actual_split_result[0], 2, &initial_SM_resources, &remaining_partition, 0, &group_params_use_case[0]);

    // STEP 3: Create a resource descriptor combining, if needed, different resources
    int groupID = 1;
    cudaDevResourceDesc_t resource_desc1;
    cudaDevResourceGenerateDesc(&resource_desc1, &actual_split_result[groupID], 1);

    cout<<"Execution Configuration:"<<endl<<"    Group: "<<groupID<<endl<<"    SM count: "<<actual_split_result[groupID].sm.smCount<<endl;
    // STEP 4: Create a green context from the descriptor, provisioning its resources
    // Create a green_ctx on GPU with current_device ID with access to resources from resource_desc
    cudaExecutionContext_t green_ctx1 {};
    cudaGreenCtxCreate(&green_ctx1, resource_desc1, current_device, 0);

    // STEP 5:  Create a stream for that green context and launch work in that stream
    cudaStream_t green_ctx_stream1;
    cudaExecutionCtxStreamCreate(&green_ctx_stream1,green_ctx1,cudaStreamDefault,0);
    
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    // long long N = 16LL * 1024LL * 1024LL;
    long long N = 128 * 20 * 1024LL * 1024LL;
    int *data;
    std::cout << "Allocating " << (N * sizeof(int)) / (1024 * 1024) << " MB of Unified Memory..." << std::endl;
    cudaMallocManaged(&data, N * sizeof(int));

    for (long long i = 0; i < N; i++)
    {
        data[i] = i;
    }
    std::cout << "CPU initialization complete. Launching kernel with 1 block, 1 thread..." << std::endl;

    cudaEventRecord(start1,green_ctx_stream1);
    accessAll<<<1, 1, 0, green_ctx_stream1>>>(data, N);
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    std::cout << "Victim kernel launched." << std::endl;
    cudaEventRecord(stop1,green_ctx_stream1);

    err = cudaEventSynchronize(stop1);
    if (err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
    }

    std::cout << "Synchronization successful." << std::endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    cout << "Victim kernel time: " << milliseconds << " ms\n";
    cudaError_t e = cudaGetLastError();
    cout << "Error: " << cudaGetErrorString(e) << endl;
    cudaFree(data);
}
