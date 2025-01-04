import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time

def compile_cuda_programs():
    # 编译两个CUDA程序
    subprocess.run(['nvcc', 'lab4_VectorAdd_Streams.cu', '-o', 'vectorAdd_streams'])
    subprocess.run(['nvcc', 'lab2_VectorAdd_Streams.cu', '-o', 'vectorAdd_normal'])

def run_test(exe_name, input_length, segment_size=None):
    start_time = time.time()
    if segment_size:
        # 运行stream版本
        cmd = [f'./{exe_name}', str(input_length), str(segment_size)]
    else:
        # 运行普通版本
        cmd = [f'./{exe_name}', str(input_length)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    return result.stdout, end_time - start_time

def main():
    # 编译程序
    compile_cuda_programs()
    
    # 测试参数
    vector_lengths = [1000000, 2000000, 5000000, 10000000]
    segment_sizes = [1000, 5000, 10000, 50000, 100000]  # 不同的段大小
    
    # 存储结果
    normal_times = []
    stream_times = {}
    
    # 测试普通版本
    for length in vector_lengths:
        _, time_taken = run_test('vectorAdd_normal', length)
        normal_times.append(time_taken)
    
    # 测试streams版本（不同段大小）
    for segment_size in segment_sizes:
        stream_times[segment_size] = []
        for length in vector_lengths:
            _, time_taken = run_test('vectorAdd_streams', length, segment_size)
            stream_times[segment_size].append(time_taken)
    
    # 绘制性能对比图
    plt.figure(figsize=(12, 6))
    plt.plot(vector_lengths, normal_times, 'o-', label='Normal Version')
    
    for segment_size in segment_sizes:
        plt.plot(vector_lengths, stream_times[segment_size], 
                'o-', label=f'Streams (segment={segment_size})')
    
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Vector Addition Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.close()
    
    # 绘制段大小影响图
    plt.figure(figsize=(12, 6))
    longest_vector = vector_lengths[-1]
    segment_performance = [stream_times[size][-1] for size in segment_sizes]
    
    plt.plot(segment_sizes, segment_performance, 'o-')
    plt.xlabel('Segment Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Impact of Segment Size (Vector Length = {longest_vector})')
    plt.grid(True)
    plt.savefig('segment_size_impact.png')

if __name__ == "__main__":
    main()
