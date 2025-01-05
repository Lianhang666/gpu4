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
    # 向量长度范围保持不变
    vector_lengths = [2048, 4096, 8192, 16384, 32768]
    
    # 固定段大小为2048
    segment_size = 2048
    
    # 存储结果
    streamed_times = []
    non_streamed_times = []
    
    # 运行测试
    for length in vector_lengths:
        # 运行非流版本
        non_stream_time = run_cuda_program('./lab2_VectorAdd_Streams', length)
        non_streamed_times.append(non_stream_time)
        print(f"\nVector length: {length}")
        print(f"Non-streamed time: {non_stream_time} ms")
        
        # 运行流版本（固定段大小为2048）
        time = run_cuda_program('./lab4_VectorAdd_Streams', length, segment_size)
        streamed_times.append(time)
        print(f"Streamed time (segment=2048): {time} ms")
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制非流版本
    plt.plot(vector_lengths, non_streamed_times, 'k-o', 
             label='Non-streamed', linewidth=2)
    
    # 绘制流版本（只有2048段大小）
    plt.plot(vector_lengths, streamed_times, 'b-o', 
             label='Streamed (segment=2048)', linewidth=2)
    
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
