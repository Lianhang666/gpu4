#include <stdio.h>
#include <stdlib.h> // For atoi and rand
#include <sys/time.h>

struct timeval start, stop;
#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < len) {                                 
        out[i] = in1[i] + in2[i];                  
    }
}

void start_timer() {
    gettimeofday(&start, NULL);
}

void stop_timer(const char* message) {
    gettimeofday(&stop, NULL);
    double elapsedTime = (stop.tv_sec - start.tv_sec) * 1000.0; // Convert seconds to milliseconds
    elapsedTime += (stop.tv_usec - start.tv_usec) / 1000.0;      // Convert microseconds to milliseconds
    printf("%s: %.2f ms\n", message, elapsedTime);
}

int main(int argc, char **argv) {
    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args
    if (argc > 1) {
        inputLength = atoi(argv[1]);
    } else {
        printf("Please provide the input length as an argument.\n");
        return 1;
    }

    printf("The input length is %d\n", inputLength);
  
    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType*)malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType*)malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType*)malloc(inputLength * sizeof(DataType));
    resultRef = (DataType*)malloc(inputLength * sizeof(DataType));
  
    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 100;
        hostInput2[i] = rand() % 100;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void**)&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc((void**)&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc((void**)&deviceOutput, inputLength * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here
    start_timer();
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    stop_timer("Time for copying data from Host to Device");

    //@@ Initialize the 1D grid and block dimensions here
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    //@@ Launch the GPU Kernel here
    start_timer();
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize(); // Ensure the kernel has completed
    stop_timer("Time for CUDA kernel execution");

    //@@ Copy the GPU memory back to the CPU here
    start_timer();
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    stop_timer("Time for copying data from Device to Host");

    //@@ Insert code below to compare the output with the reference
    bool match = true;
    for (int i = 0; i < inputLength; i++) {
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU result = %f, CPU result = %f\n", i, hostOutput[i], resultRef[i]);
            match = false;
            break;
        }
    }
    if (match) {
        printf("Results match.\n");
    } else {
        printf("Results do not match.\n");
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
