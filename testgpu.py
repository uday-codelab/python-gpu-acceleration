import cupy as cp
import numpy as np
import time

def benchmark_matrix_mul(N=4000, dtype=np.float32):
    print(f"\n--- Benchmarking {N}x{N} Matrix (Type: {dtype}) ---")

    # 1. CPU EXECUTION
    a_cpu = np.random.rand(N, N).astype(dtype)
    b_cpu = np.random.rand(N, N).astype(dtype)
    
    start_cpu = time.time()
    res_cpu = np.dot(a_cpu, b_cpu)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"CPU Time: {cpu_time:.4f} seconds")

    # 2. GPU EXECUTION (Includes Data Transfer)
    start_gpu_total = time.time()
    
    # Move data to VRAM
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    
    # Compute
    res_gpu = cp.dot(a_gpu, b_gpu)
    
    # IMPORTANT: Wait for GPU to finish before stopping the clock
    cp.cuda.Stream.null.synchronize() 
    
    end_gpu_total = time.time()
    gpu_time = end_gpu_total - start_gpu_total
    print(f"GPU Time (w/ transfer): {gpu_time:.4f} seconds")

    # 3. SPEEDUP CALCULATION
    print(f"Result: GPU is {cpu_time / gpu_time:.2f}x faster than CPU")

if __name__ == "__main__":
    # Test different precisions
    benchmark_matrix_mul(N=4000, dtype=np.float32) # Single precision (Fast)
    benchmark_matrix_mul(N=4000, dtype=np.float64) # Double precision (Precise)
