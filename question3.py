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
    # 正确的向量长度
    input_length = 32768
    
    # 不同的段大小
    segment_sizes = [2048, 4096, 8192, 16384]
    
    # 运行测试
    execution_times = []
    for size in segment_sizes:
        time = run_with_segment_size(input_length, size)
        execution_times.append(time)
        print(f"Vector length: {input_length}, Segment size: {size}, Time: {time} ms")
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(segment_sizes, execution_times, 'b-o', linewidth=2)
    plt.xlabel('Segment Size')
    plt.ylabel('Execution Time (ms)')
    plt.title(f'Impact of Segment Size on Performance\n(Vector Length: {input_length})')
    plt.grid(True)
    
    # 在数据点上添加标签
    for i, (size, time) in enumerate(zip(segment_sizes, execution_times)):
        plt.annotate(f'Size: {size}\nTime: {time:.2f}ms', 
                    (size, time),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.savefig('segment_size_analysis.png')
    plt.close()

if __name__ == "__main__":
    print("Starting segment size analysis...")
    analyze_segment_sizes()
    print("\nSegment size analysis completed. Check segment_size_analysis.png")
