import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_cuda_program(program_path, input_length, segment_size=None):
    if segment_size:
        cmd = [program_path, str(input_length), str(segment_size)]
    else:
        cmd = [program_path, str(input_length)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if "Total execution time with streams" in line:
                return float(line.split(':')[1].strip().split()[0])
            elif "Time for CUDA kernel execution" in line:
                return float(line.split(':')[1].strip().split()[0])
    except Exception as e:
        print(f"Error running CUDA program: {e}")
        return None

def compare_performance():
    # Vector lengths to test
    vector_lengths = [1000000, 5000000, 10000000, 50000000]
    segment_size = 1000000  # Fixed segment size for streamed version
    
    # Store results
    streamed_times = []
    non_streamed_times = []
    
    # Run tests
    for length in vector_lengths:
        # Run streamed version
        stream_time = run_cuda_program('./vecAdd_stream', length, segment_size)
        streamed_times.append(stream_time)
        
        # Run non-streamed version
        non_stream_time = run_cuda_program('./vecAdd_non_stream', length)
        non_streamed_times.append(non_stream_time)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(vector_lengths, non_streamed_times, 'b-o', label='Non-streamed')
    plt.plot(vector_lengths, streamed_times, 'r-o', label='Streamed')
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison: Streamed vs Non-streamed Vector Addition')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    compare_performance()
    print("Performance comparison completed. Check performance_comparison.png")
