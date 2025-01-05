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
    # Test different vector lengths
    vector_lengths = [1000000, 5000000, 10000000, 50000000]
    
    # Test different segment size ratios
    segment_ratios = [4, 8, 16, 32]  # Will divide vector length by these numbers
    
    plt.figure(figsize=(12, 8))
    
    # Plot non-streamed version
    non_streamed_times = []
    for length in vector_lengths:
        time = run_cuda_program('./lab2_VectorAdd_Streams', length)
        non_streamed_times.append(time)
        print(f"\nVector length: {length}")
        print(f"Non-streamed time: {time} ms")
    
    plt.plot(vector_lengths, non_streamed_times, 'k-o', label='Non-streamed', linewidth=2)
    
    # Test different segment sizes
    for ratio in segment_ratios:
        streamed_times = []
        for length in vector_lengths:
            segment_size = length // ratio
            time = run_cuda_program('./lab4_VectorAdd_Streams', length, segment_size)
            streamed_times.append(time)
            print(f"Streamed time (segment size = 1/{ratio}): {time} ms")
        
        plt.plot(vector_lengths, streamed_times, 
                marker='o', 
                label=f'Streamed (segment=length/{ratio})')
    
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison: Streamed vs Non-streamed Vector Addition')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    print("Starting performance comparison...")
    print("This will test multiple segment sizes for the streamed version.")
    compare_performance()
    print("\nPerformance comparison completed. Check performance_comparison.png")
