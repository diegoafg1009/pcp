#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to compute local dot product
float compute_dot_product(float *a, float *b, size_t size) {
    float local_sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        local_sum += a[i] * b[i];
    }
    return local_sum;
}

void print_cpu_info() {
    FILE *fp;
    char buffer[128];
    fp = popen("cat /proc/cpuinfo | grep 'model name' | uniq", "r");
    if (fp == NULL) {
        printf("Failed to get CPU info\n");
        return;
    }
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        printf("%s", buffer);
    }
    pclose(fp);
}

int main(int argc, char *argv[]) {
    int rank, size;
    size_t vector_size = 100000000;  // Same size as other implementations
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate local size for each process
    size_t local_size = vector_size / size;
    if (rank == size - 1) {
        local_size += vector_size % size;  // Last process takes remaining elements
    }
    
    // Allocate local arrays
    float *a = (float*)malloc(local_size * sizeof(float));
    float *b = (float*)malloc(local_size * sizeof(float));
    
    if (!a || !b) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize local arrays
    for (size_t i = 0; i < local_size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Start timing
    double start_time = MPI_Wtime();
    
    // Compute local dot product using the separate function
    float local_sum = compute_dot_product(a, b, local_size);
    
    // Reduce results across all processes
    float global_sum = 0.0f;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // End timing
    double end_time = MPI_Wtime();
    double execution_time = (end_time - start_time) * 1000.0;
    
    // Print results from root process
    if (rank == 0) {
        double throughput = (2.0 * vector_size * sizeof(float)) / (execution_time * 1000000.0);
        printf("MPI Dot Product Implementation\n");
        print_cpu_info();
        printf("Number of processes: %d\n", size);
        printf("Vector size: %zu elements (%.2f GB per vector)\n", 
               vector_size, (vector_size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
        printf("\nResults and Timing:\n");
        printf("Dot product = %.1f\n", global_sum);
        printf("Execution time: %.2f ms\n", execution_time);
        printf("Throughput: %.2f GB/s\n", throughput);
    }
    
    // Cleanup
    free(a);
    free(b);
    MPI_Finalize();
    
    return 0;
}