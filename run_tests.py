import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time

def compile_cuda_programs():
    subprocess.run(['nvcc', 'lab4_VectorAdd_Streams.cu', '-o', 'vectorAdd_streams'])
    subprocess.run(['nvcc', 'lab2_VectorAdd_Streams.cu', '-o', 'vectorAdd_normal'])

def run_nvprof(exe_name, input_length, segment_size=None):
    # 使用nvprof收集性能数据
    output_file = f'nvprof_{exe_name}_{input_length}.nvvp'
    if segment_size:
        cmd = ['nvprof', '--output-profile', output_file, 
               f'./{exe_name}', str(input_length), str(segment_size)]
    else:
        cmd = ['nvprof', '--output-profile', output_file, 
               f'./{exe_name}', str(input_length)]
    
    subprocess.run(cmd)
    return output_file

def run_performance_tests():
    vector_lengths = [1000000, 2000000, 5000000, 10000000]
    segment_sizes = [1000, 5000, 10000, 50000, 100000]
    
    # 存储结果
    normal_times = []
    stream_times = {}
    
    # 测试普通版本和生成nvprof数据
    print("Testing normal version...")
    for length in vector_lengths:
        run_nvprof('vectorAdd_normal', length)
        # 运行实际测试并记录时间
        cmd = ['./vectorAdd_normal', str(length)]
        start = time.time()
        subprocess.run(cmd)
        normal_times.append(time.time() - start)
    
    # 测试streams版本
    print("Testing streams version...")
    for segment_size in segment_sizes:
        stream_times[segment_size] = []
        for length in vector_lengths:
            run_nvprof('vectorAdd_streams', length, segment_size)
            # 运行实际测试并记录时间
            cmd = ['./vectorAdd_streams', str(length), str(segment_size)]
            start = time.time()
            subprocess.run(cmd)
            stream_times[segment_size].append(time.time() - start)
    
    return vector_lengths, normal_times, stream_times

def plot_performance_comparison(vector_lengths, normal_times, stream_times):
    plt.figure(figsize=(12, 6))
    plt.plot(vector_lengths, normal_times, 'o-', label='Normal Version')
    
    for segment_size, times in stream_times.items():
        plt.plot(vector_lengths, times, 'o-', 
                label=f'Streams (segment={segment_size})')
        
        # 计算并显示性能提升
        speedup = [n/s for n, s in zip(normal_times, times)]
        print(f"\nSpeedup for segment size {segment_size}:")
        for length, sp in zip(vector_lengths, speedup):
            print(f"Vector length {length}: {sp:.2f}x speedup")
    
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Vector Addition Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    print("\nPerformance comparison plot saved as 'performance_comparison.png'")

def plot_segment_size_impact(segment_sizes, stream_times):
    plt.figure(figsize=(12, 6))
    # 使用最大向量长度的结果
    longest_vector_times = [times[-1] for times in stream_times.values()]
    
    plt.plot(segment_sizes, longest_vector_times, 'o-')
    plt.xlabel('Segment Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Impact of Segment Size on Performance')
    plt.grid(True)
    plt.savefig('segment_size_impact.png')
    print("\nSegment size impact plot saved as 'segment_size_impact.png'")

def main():
    print("Compiling CUDA programs...")
    compile_cuda_programs()
    
    print("\nRunning performance tests...")
    vector_lengths, normal_times, stream_times = run_performance_tests()
    
    print("\nGenerating plots...")
    plot_performance_comparison(vector_lengths, normal_times, stream_times)
    plot_segment_size_impact(list(stream_times.keys()), stream_times)
    
    print("\nNVProf data has been collected. You can now use NVIDIA Visual Profiler (nvvp)")
    print("to analyze the trace files generated for each run.")

if __name__ == "__main__":
    main()
