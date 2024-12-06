#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

// Structure to pass data to threads
typedef struct {
    float *a;
    float *b;
    size_t start;
    size_t end;
    float partial_result;
} ThreadData;

// Function that each thread will execute
void* dot_product_thread(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    float sum = 0.0f;
    
    // Compute partial dot product
    for (size_t i = data->start; i < data->end; i++) {
        sum += data->a[i] * data->b[i];
    }
    
    data->partial_result = sum;
    pthread_exit(NULL);
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
    // Set size to 100 million elements
    size_t size = 100000000;
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);  // Use all available CPU cores
    
    // Print basic information
    printf("CPU Dot Product Implementation (Pthreads)\n");
    print_cpu_info();
    printf("Vector size: %zu elements (%.2f GB per vector)\n", 
           size, (size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
    printf("Number of threads: %d\n\n", num_threads);
    
    // Allocate and initialize vectors
    float *a = (float*)malloc(size * sizeof(float));
    float *b = (float*)malloc(size * sizeof(float));
    
    if (!a || !b) {
        printf("Memory allocation failed!\n");
        return -1;
    }
    
    // Initialize vectors
    for (size_t i = 0; i < size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Create timing variables
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // Create thread data structures
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));
    
    // Calculate elements per thread
    size_t elements_per_thread = size / num_threads;
    
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].start = i * elements_per_thread;
        thread_data[i].end = (i == num_threads - 1) ? size : (i + 1) * elements_per_thread;
        thread_data[i].partial_result = 0.0f;
        
        if (pthread_create(&threads[i], NULL, dot_product_thread, &thread_data[i])) {
            printf("Error creating thread %d\n", i);
            return -1;
        }
    }
    
    // Wait for all threads to complete
    float final_result = 0.0f;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        final_result += thread_data[i].partial_result;
    }
    
    // Calculate execution time
    gettimeofday(&end_time, NULL);
    double execution_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                          (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    
    // Calculate throughput
    double throughput = (2.0 * size * sizeof(float)) / (execution_time * 1000000.0);  // GB/s
    
    // Print results
    printf("Results and Timing:\n");
    printf("Dot product = %.1f\n", final_result);
    printf("Execution time: %.2f ms\n", execution_time);
    printf("Throughput: %.2f GB/s\n", throughput);
    
    // Cleanup
    free(threads);
    free(thread_data);
    free(a);
    free(b);
    
    return 0;
} 