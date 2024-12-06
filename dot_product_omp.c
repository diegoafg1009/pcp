#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

float compute_dot_product(float *a, float *b, size_t size) {
    float result = 0.0f;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
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

int main() {
    size_t size = 100000000;
    
    printf("OpenMP Dot Product Implementation\n");
    print_cpu_info();
    printf("Number of OpenMP threads: %d\n", omp_get_max_threads());
    printf("Vector size: %zu elements (%.2f GB per vector)\n", 
           size, (size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
    
    // Allocate vectors
    float *a = (float*)malloc(size * sizeof(float));
    float *b = (float*)malloc(size * sizeof(float));
    
    if (!a || !b) {
        printf("Memory allocation failed!\n");
        return -1;
    }
    
    // Initialize vectors in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    double start_time = omp_get_wtime();
    
    // Compute dot product using the separate function
    float result = compute_dot_product(a, b, size);
    
    double end_time = omp_get_wtime();
    double execution_time = (end_time - start_time) * 1000.0;
    
    double throughput = (2.0 * size * sizeof(float)) / (execution_time * 1000000.0);
    
    printf("\nResults and Timing:\n");
    printf("Dot product = %.1f\n", result);
    printf("Execution time: %.2f ms\n", execution_time);
    printf("Throughput: %.2f GB/s\n", throughput);
    
    free(a);
    free(b);
    
    return 0;
}