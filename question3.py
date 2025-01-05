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
    # 正确的向量长度
    vector_lengths = [2048]
    
    # 不同的段大小
    segment_sizes = [2048, 4096, 8192, 16384]
    
    # 存储结果
    streamed_times = {size: [] for size in segment_sizes}
    non_streamed_times = []
    
    # 运行测试
    for length in vector_lengths:
        # 运行非流版本
        non_stream_time = run_cuda_program('./lab2_VectorAdd_Streams', length)
        non_streamed_times.append(non_stream_time)
        print(f"\nVector length: {length}")
        print(f"Non-streamed time: {non_stream_time} ms")
        
        # 运行不同段大小的流版本
        for seg_size in segment_sizes:
            if seg_size <= length:  # 只在段大小小于等于向量长度时测试
                time = run_cuda_program('./lab4_VectorAdd_Streams', length, seg_size)
                streamed_times[seg_size].append(time)
                print(f"Streamed time (segment={seg_size}): {time} ms")
            else:
                streamed_times[seg_size].append(None)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制非流版本
    plt.plot(vector_lengths, non_streamed_times, 'k-o', 
             label='Non-streamed', linewidth=2)
    
    # 绘制不同段大小的流版本
    colors = ['b', 'g', 'r', 'c']
    for seg_size, color in zip(segment_sizes, colors):
        valid_points = [(x, y) for x, y in zip(vector_lengths, streamed_times[seg_size]) if y is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, f'{color}-o', 
                    label=f'Streamed (segment={seg_size})')
    
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison: Streamed vs Non-streamed Vector Addition')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    print("Starting performance comparison...")
    compare_performance()
    print("\nPerformance comparison completed. Check performance_comparison.png")
