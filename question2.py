import subprocess
import os

def run_nsys(cuda_program, input_length, segment_size):
    """Run nsys profile with specific parameters"""
    output_name = "report"  # 固定输出文件名
    cmd = [
        "nsys", "profile",
        "--stats=true",
        "--force-overwrite=true",
        "--output", output_name,
        cuda_program,
        str(input_length),
        str(segment_size)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nProfile generated for input length {input_length}, segment size {segment_size}")
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

def analyze_single_scenario():
    # 固定的测试场景
    vector_length = 8192    # 中等大小的向量
    segment_size = 4096     # 段大小为向量长度的一半
    
    print(f"\nAnalyzing vector addition...")
    print(f"Vector length: {vector_length}")
    print(f"Segment size: {segment_size}")
    
    run_nsys("./lab4_VectorAdd_Streams", vector_length, segment_size)
    
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
   - Stream overlap patterns
   - Memory transfer efficiency
   - Kernel execution parallelism
""")

if __name__ == "__main__":
    print("Starting NSYS analysis...")
    clean_old_report()
    analyze_single_scenario()
    print("\nNSYS analysis completed. Use nsys-ui report.nsys-rep to view the visualization.")
