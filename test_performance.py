import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time

def compile_cuda_programs():
    """
    编译 CUDA 程序：分别生成
      - vectorAdd_streams (多流版本)
      - vectorAdd_normal  (普通版本)
    需保证 lab4_VectorAdd_Streams.cu 和 lab2_VectorAdd_Streams.cu 存在
    或根据实际文件名进行调整。
    """
    subprocess.run(['nvcc', 'lab4_VectorAdd_Streams.cu', '-o', 'vectorAdd_streams'])
    subprocess.run(['nvcc', 'lab2_VectorAdd_Streams.cu', '-o', 'vectorAdd_normal'])

def run_nsys(exe_name, input_length, segment_size=None):
    """
    使用 Nsight Systems (nsys) 收集性能数据并生成 .qdrep 文件。
    如果不需要收集 trace，可注释掉此函数的调用。
    """
    output_prefix = f'nsys_{exe_name}_{input_length}'
    # 如果有 segment_size，需要拼在命令行里
    if segment_size:
        cmd = [
            'nsys', 'profile',
            '-o', output_prefix,       # 输出文件前缀，会生成 .qdrep 和 .sqlite
            '--trace=cuda',           # 抓取 CUDA trace，用于查看 stream/内存拷贝时间线
            f'./{exe_name}', 
            str(input_length), 
            str(segment_size)
        ]
    else:
        cmd = [
            'nsys', 'profile',
            '-o', output_prefix,
            '--trace=cuda',
            f'./{exe_name}', 
            str(input_length)
        ]

    print(f"Collecting Nsight Systems trace with command:\n{' '.join(cmd)}\n")
    subprocess.run(cmd)
    # 返回 .qdrep 文件路径（Nsight Systems GUI 可打开）
    return output_prefix + '.qdrep'

def run_performance_tests():
    """
    分别对 normal 和 streams 版本在若干 vector_lengths 上测试，
    streams 版本会在不同 segment_sizes 下重复测试。
    同时收集时间，并可调用 run_nsys() 收集性能追踪文件。
    """
    # 你可以根据需要调整测试规模
    vector_lengths = [1000000, 2000000, 5000000, 10000000]
    segment_sizes = [1000, 5000, 10000, 50000, 100000]
    
    # 用于存储普通版本（无流）的总执行时间
    normal_times = []
    # 用于存储多流版本在不同 segment_size 下的执行时间
    stream_times = {}
    
    # 测试普通版本
    print("Testing normal version...")
    for length in vector_lengths:
        # 收集 Nsight Systems trace（若不需要，可注释）
        run_nsys('vectorAdd_normal', length)
        
        # 实际运行并记录 wall-clock 时间
        cmd = ['./vectorAdd_normal', str(length)]
        start = time.time()
        subprocess.run(cmd)
        elapsed = time.time() - start
        normal_times.append(elapsed)
        print(f"Normal version (length={length}) took {elapsed:.4f} s\n")

    # 测试多流版本
    print("Testing streams version...")
    for seg_size in segment_sizes:
        stream_times[seg_size] = []
        for length in vector_lengths:
            run_nsys('vectorAdd_streams', length, seg_size)
            
            cmd = ['./vectorAdd_streams', str(length), str(seg_size)]
            start = time.time()
            subprocess.run(cmd)
            elapsed = time.time() - start
            stream_times[seg_size].append(elapsed)
            print(f"Streams version (length={length}, segment={seg_size}) took {elapsed:.4f} s\n")
    
    return vector_lengths, normal_times, stream_times

def plot_performance_comparison(vector_lengths, normal_times, stream_times):
    """
    绘制普通版本与多流版本（不同 segment_size）在不同向量长度下的执行时间对比图。
    并打印/计算加速比。
    """
    plt.figure(figsize=(12, 6))
    # 普通版本的执行时间曲线
    plt.plot(vector_lengths, normal_times, 'o-', label='Normal Version')
    
    # 多流版本在不同 segment_size 下的执行时间曲线
    for segment_size, times in stream_times.items():
        plt.plot(vector_lengths, times, 'o-', label=f'Streams (segment={segment_size})')
        
        # 计算并打印 speedup
        speedup = [n / s for n, s in zip(normal_times, times)]
        print(f"\nSpeedup for segment size {segment_size}:")
        for length, sp in zip(vector_lengths, speedup):
            print(f"  Vector length {length}: {sp:.2f}x speedup")
    
    plt.xlabel('Vector Length')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Vector Addition Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    print("\nPerformance comparison plot saved as 'performance_comparison.png'")

def plot_segment_size_impact(segment_sizes, stream_times):
    """
    在固定（或选取最大）向量长度下，查看不同 segment_size 对执行时间的影响。
    这里示例只取 stream_times[segment_size] 的最后一个值(对应 vector_lengths[-1])，
    也可以根据需要再做更多细分。
    """
    plt.figure(figsize=(12, 6))
    # 使用在最大 vector_length 条件下的执行时间
    # stream_times[seg_size] 是一个list，对应 len(vector_lengths) 个结果，取最后一个
    longest_vector_times = [times[-1] for times in stream_times.values()]
    
    plt.plot(segment_sizes, longest_vector_times, 'o-')
    plt.xlabel('Segment Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Impact of Segment Size on Performance (largest vector)')
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
    
    print("\nTrace data has been collected via Nsight Systems (nsys).")
    print("You can now open the generated '.qdrep' files in Nsight Systems GUI")
    print("to analyze the timeline for each run.")

if __name__ == "__main__":
    main()
