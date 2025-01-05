import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def run_nsys(cuda_program, input_length, segment_size):
    """Run nsys and save the output"""
    report_file = "report"
    cmd = [
        "nsys", "profile",
        "--stats=true",
        "--output", report_file,
        cuda_program,
        str(input_length),
        str(segment_size)
    ]
    
    subprocess.run(cmd)
    return report_file

def analyze_timeline():
    # Run nsys profile
    input_length = 10000000
    segment_size = 1000000
    report_file = run_nsys("./lab4_VectorAdd_Streams", input_length, segment_size)
    
    print("""
Analysis Instructions:
1. The nsys profile has been generated and saved
2. To view the timeline visualization:
   - Use: nsys-ui report.nsys-rep
   - This will open the NVIDIA Nsight Systems UI
3. In the UI, you can:
   - See the CUDA operations timeline
   - Analyze kernel execution and memory transfers
   - Look for overlap between computation and communication
4. Key things to look for:
   - Multiple stream operations running concurrently
   - Overlap between memory transfers and kernel execution
   - The effect of different streams on parallelization
""")

if __name__ == "__main__":
    analyze_timeline()
    print("NSYS analysis completed. Use nsys-ui to view the timeline visualization.")
