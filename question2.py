import subprocess
import os

def run_nsys(cuda_program, vector_length):
    """Run nsys profile with specific parameters"""
    output_name = "report"  # 固定输出文件名
    cmd = [
        "nsys", "profile",
        "--stats=true",
        "--force-overwrite=true",
        "--output", output_name,
        cuda_program,
        str(vector_length)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nProfile generated for vector length {vector_length}")
    except subprocess.CalledProcessError as e:
        print(f"Error running nsys profile: {e}")

def clean_old_report():
    """Clean up old report file if it exists"""
    if os.path.exists('report.nsys-rep'):
        try:
            os.remove('report.nsys-rep')
            print("Removed old report file")
        except OSError as e:
            print(f"Error removing old report: {e}")

def analyze_stream_performance():
    # 使用较大的向量长度来更好地展示流的效果
    vector_length = 1048576  # 使用1M大小的向量
    
    print(f"\nAnalyzing vector addition with streams...")
    print(f"Vector length: {vector_length}")
    print(f"Number of streams: 4")  # 由CUDA代码中的NUM_STREAMS定义
    
    # 运行性能分析
    run_nsys("./lab4_VectorAdd_Streams", vector_length)
    
    print("""
Analysis Instructions:
1. A profile report has been generated (report.nsys-rep)

2. To view the timeline visualization:
   - Use: nsys-ui report.nsys-rep

3. In the Nsight Systems UI, look for:
   - CUDA API calls timeline
   - Kernel execution timeline
   - Memory operations timeline
   - Stream synchronization points

4. Key aspects to analyze:
   - H2D memory transfers (green blocks)
   - Kernel execution (blue blocks)
   - D2H memory transfers (red blocks)
   - Stream overlap patterns
   - Multiple operations executing concurrently
   - Memory transfer efficiency

5. Expected pattern:
   - Should see overlapping of memory transfers and kernel execution
   - Multiple streams operating concurrently
   - Efficient utilization of hardware resources
""")

if __name__ == "__main__":
    print("Starting NSYS analysis...")
    clean_old_report()
    analyze_stream_performance()
    print("\nNSYS analysis completed. Use nsys-ui report.nsys-rep to view the visualization.")
