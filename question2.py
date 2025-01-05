import subprocess
import json
import os

def run_nsys(cuda_program, input_length, segment_size, output_name):
    """Run nsys profile with specific parameters"""
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

def analyze_scenarios():
    # Test scenarios
    scenarios = [
        {
            "name": "small_vector",
            "length": 2048,
            "segment": 2048
        },
        {
            "name": "medium_vector",
            "length": 8192,
            "segment": 4096
        },
        {
            "name": "large_vector",
            "length": 32768,
            "segment": 8192
        }
    ]
    
    # Create profiles for each scenario
    for scenario in scenarios:
        print(f"\nAnalyzing {scenario['name']}...")
        output_name = f"report_{scenario['name']}"
        run_nsys("./lab4_VectorAdd_Streams", 
                scenario['length'], 
                scenario['segment'],
                output_name)

    print("""
Analysis Instructions:
1. Multiple profile reports have been generated:
   - report_small_vector.nsys-rep  (Vector Length: 2048, Segment: 2048)
   - report_medium_vector.nsys-rep (Vector Length: 8192, Segment: 4096)
   - report_large_vector.nsys-rep  (Vector Length: 32768, Segment: 8192)

2. To view the timeline visualizations:
   - Use: nsys-ui report_[scenario_name].nsys-rep
   - Example: nsys-ui report_small_vector.nsys-rep

3. In the Nsight Systems UI, look for:
   - CUDA API calls timeline
   - Kernel execution timeline
   - Memory operations timeline
   - Stream synchronization points

4. Key aspects to analyze:
   - Stream overlap patterns
   - Memory transfer efficiency
   - Kernel execution parallelism
   - Impact of different segment sizes on parallelization

5. Compare different scenarios:
   - How parallelism scales with vector size
   - Effect of segment size on stream utilization
   - Memory transfer patterns across different sizes
   
6. Performance factors to consider:
   - Memory transfer overhead vs computation time
   - Stream synchronization overhead
   - Efficiency of parallelization
""")

def clean_old_reports():
    """Clean up old report files if they exist"""
    for file in os.listdir('.'):
        if file.startswith('report_') and file.endswith('.nsys-rep'):
            try:
                os.remove(file)
                print(f"Removed old report: {file}")
            except OSError as e:
                print(f"Error removing {file}: {e}")

if __name__ == "__main__":
    print("Starting NSYS analysis...")
    print("Cleaning up old reports...")
    clean_old_reports()
    analyze_scenarios()
    print("\nNSYS analysis completed. Use nsys-ui to view the generated reports.")
