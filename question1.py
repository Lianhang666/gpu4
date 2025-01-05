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
    # 向量长度范围
    vector_lengths = [2000, 5000, 10000, 500000, 1000000]
    
    # 存储结果
    streamed_times = []
    non_streamed_times = []
    segment_sizes = []  # 记录每个向量长度对应的段大小
    
    # 运行测试
    for length in vector_lengths:
        # 计算当前向量长度的合适段大小（向量长度除以流的数量）
        segment_size = length // 4  # 4 是 NUM_STREAMS 的值
        segment_sizes.append(segment_size)
        
        # 运行非流版本
        non_stream_time = run_cuda_program('./lab2_VectorAdd_Streams', length)
        non_streamed_times.append(non_stream_time)
        print(f"\nVector length: {length}")
        print(f"Non-streamed time: {non_stream_time} ms")
        
        # 运行流版本（使用计算出的段大小）
        time = run_cuda_program('./lab4_VectorAdd_Streams', length)
        streamed_times.append(time)
        print(f"Streamed time (segment={segment_size}): {time} ms")
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制非流版本
    plt.plot(vector_lengths, non_streamed_times, 'k-o', 
             label='Non-streamed', linewidth=2)
    
    # 绘制流版本
    plt.plot(vector_lengths, streamed_times, 'b-o', 
             label='Streamed (auto segment)', linewidth=2)
    
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison: Streamed vs Non-streamed Vector Addition')
    
    # 在图表上添加段大小信息
    for i, (length, time, seg_size) in enumerate(zip(vector_lengths, streamed_times, segment_sizes)):
        plt.annotate(f'seg={seg_size}', 
                    (length, time),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
    
    plt.legend()
    plt.grid(True)
    plt.xscale('log')  # 使用对数刻度更好地显示不同量级的向量长度
    plt.yscale('log')  # 使用对数刻度更好地显示执行时间差异
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    print("Starting performance comparison...")
    compare_performance()
    print("\nPerformance comparison completed. Check performance_comparison.png")
