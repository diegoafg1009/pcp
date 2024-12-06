import subprocess
import re
import os

def run_program(command, keyword, runs=5):
    """Run a shell command multiple times and return the average execution time."""
    total_time = 0.0
    for _ in range(runs):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        time = extract_time(output, keyword)
        if time is not None:
            total_time += time
        else:
            print(f"Failed to extract time from output:\n{output}")
            return None
    return total_time / runs

def extract_time(output, keyword):
    """Extract execution time from the program output."""
    match = re.search(rf"{keyword}:\s+([\d.]+)\s+ms", output)
    return float(match.group(1)) if match else None

def main():
    # Compile all implementations
    print("Compiling implementations...")
    os.system("nvcc dot_product.cu -o dot_product_cuda -O3")
    os.system("gcc -o dot_product_pthread dot_product_pthread.c -pthread -O3")
    os.system("gcc -o dot_product_omp dot_product_omp.c -fopenmp -O3")
    os.system("mpicc dot_product_mpi.c -o dot_product_mpi -O3")

    # Define commands to run each program
    commands = {
        "CUDA": "./dot_product_cuda",
        "Pthreads": "./dot_product_pthread",
        "OpenMP": "./dot_product_omp",
        "MPI": f"mpirun --use-hwthread-cpus ./dot_product_mpi"
    }

    # Dictionary to store average execution times
    average_times = {}

    # Run each program and capture average execution times
    for name, command in commands.items():
        print(f"Running {name} implementation...")
        if name == "CUDA":
            avg_time = run_program(command, "Kernel Execution")
        else:
            avg_time = run_program(command, "Execution time")
        if avg_time is not None:
            average_times[name] = avg_time
            print(f"Average execution time for {name}: {avg_time:.2f} ms\n")

    # Calculate speedup ratios
    max_time = max(average_times.values())
    print("\nSpeedup Ratios (relative to slowest):")
    for name, time in average_times.items():
        speedup = max_time / time
        print(f"{name}: {speedup:.2f}x")

    # Determine the fastest implementation
    fastest = min(average_times, key=average_times.get)
    print(f"\nFastest implementation: {fastest}")

    # Print the dictionary of average times
    print("\nAverage Execution Times:")
    for name, time in average_times.items():
        print(f"{name}: {time:.2f} ms")

if __name__ == "__main__":
    main()
