import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_with_segment_size(input_length, segment_size):
    cmd = ["./lab4_VectorAdd_Streams", str(input_length), str(segment_size)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if "Total execution time with streams" in line:
                return float(line.split(':')[1].strip().split()[0])
    except Exception as e:
        print(f"Error running CUDA program: {e}")
        return None

def analyze_segment_sizes():
    # Fixed large vector length
    input_length = 50000000
    
    # Different segment sizes to test (powers of 2)
    segment_sizes = [
        input_length // 32,
        input_length // 16,
        input_length // 8,
        input_length // 4,
        input_length // 2,
        input_length
    ]
    
    # Run tests for each segment size
    execution_times = []
    for size in segment_sizes:
        time = run_with_segment_size(input_length, size)
        execution_times.append(time)
        print(f"Segment size: {size}, Time: {time} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(segment_sizes, execution_times, 'b-o')
    plt.xlabel('Segment Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('Impact of Segment Size on Performance')
    plt.grid(True)
    plt.xscale('log')
    
    # Add segment size labels on data points
    for i, size in enumerate(segment_sizes):
        plt.annotate(f'{size}', 
                    (size, execution_times[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.savefig('segment_size_analysis.png')
    plt.close()

if __name__ == "__main__":
    analyze_segment_sizes()
    print("Segment size analysis completed. Check segment_size_analysis.png")
