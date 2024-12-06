#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for dot product
__global__ void dotProduct(float *a, float *b, float *c, int size) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0;
    // Each thread computes part of the dot product
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // Store in shared memory
    cache[cacheIndex] = temp;
    
    // Synchronize threads
    __syncthreads();
    
    // Reduce within block
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    
    // Write result for this block to global memory
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    
    // Print hardware specifications
    printf("CUDA Device Information:\n");
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total global memory: %.2f GB\n\n", 
           (float)prop.totalGlobalMem / (1024 * 1024 * 1024));

    // Set  size to 100 million elements
    size_t size = 100000000;  
    printf("\nVector size: %zu elements (%.2f GB per vector)\n", 
           size, (size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
    
    float *a, *b;     // Host vectors
    float *d_a, *d_b, *d_c;  // Device vectors
    float result = 0;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing total operation
    cudaEventRecord(start);
    
    // Allocate host memory with pinned memory for better transfer performance
    cudaMallocHost(&a, size * sizeof(float));
    cudaMallocHost(&b, size * sizeof(float));
    
    // Initialize vectors
    #pragma omp parallel for  // Optional: Use OpenMP for faster initialization
    for (size_t i = 0; i < size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_c, ((size + 255)/256) * sizeof(float));
    
    // Copy data to device and measure transfer time
    cudaEvent_t transfer_start, transfer_stop;
    cudaEventCreate(&transfer_start);
    cudaEventCreate(&transfer_stop);
    
    cudaEventRecord(transfer_start);
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(transfer_stop);
    cudaEventSynchronize(transfer_stop);
    
    float transfer_time = 0;
    cudaEventElapsedTime(&transfer_time, transfer_start, transfer_stop);
    
    // Launch kernel with timing
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    cudaEventRecord(kernel_start);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    dotProduct<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    
    // Allocate host memory for partial results
    float *partial_c;
    cudaMallocHost(&partial_c, numBlocks * sizeof(float));
    
    // Copy results back and measure
    cudaEvent_t transfer_back_start, transfer_back_stop;
    cudaEventCreate(&transfer_back_start);
    cudaEventCreate(&transfer_back_stop);
    
    cudaEventRecord(transfer_back_start);
    cudaMemcpy(partial_c, d_c, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(transfer_back_stop);
    cudaEventSynchronize(transfer_back_stop);
    
    float transfer_back_time = 0;
    cudaEventElapsedTime(&transfer_back_time, transfer_back_start, transfer_back_stop);
    
    // Sum up partial results
    for (int i = 0; i < numBlocks; i++) {
        result += partial_c[i];
    }
    
    // Get total time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);
    
    // Print detailed timing results
    printf("\nResults and Timing:\n");
    printf("Dot product = %.1f\n", result);
    printf("Host to Device Transfer: %.2f ms\n", transfer_time);
    printf("Kernel Execution: %.2f ms\n", kernel_time);
    printf("Device to Host Transfer: %.2f ms\n", transfer_back_time);
    printf("Total Time: %.2f ms\n", total_time);
    
    // Calculate throughput
    float throughput = (2.0f * size * sizeof(float)) / (kernel_time * 1000000.0f);  // GB/s
    printf("Kernel Throughput: %.2f GB/s\n", throughput);
    
    // Cleanup
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(partial_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(transfer_start);
    cudaEventDestroy(transfer_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(transfer_back_start);
    cudaEventDestroy(transfer_back_stop);
    
    return 0;
} 