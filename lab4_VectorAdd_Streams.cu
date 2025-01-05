#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval start, stop;
#define DataType double
#define NUM_STREAMS 4

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
    double elapsedTime = (stop.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (stop.tv_usec - start.tv_usec) / 1000.0;
    printf("%s: %.2f ms\n", message, elapsedTime);
}

int main(int argc, char **argv) {
    int inputLength, segmentSize;
    DataType *hostInput1, *hostInput2, *hostOutput, *resultRef;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;

    // Read input length and segment size
    if (argc > 2) {
        inputLength = atoi(argv[1]);
        segmentSize = atoi(argv[2]);
    } else {
        printf("Usage: %s <input_length> <segment_size>\n", argv[0]);
        return 1;
    }
    printf("The input length is %d, segment size is %d\n", inputLength, segmentSize);

    // Allocate host memory - using same allocation as lab2
    hostInput1 = (DataType*)malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType*)malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType*)malloc(inputLength * sizeof(DataType));
    resultRef = (DataType*)malloc(inputLength * sizeof(DataType));

    // Initialize input arrays - same as lab2
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 100;
        hostInput2[i] = rand() % 100;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    // Allocate GPU memory - same as lab2
    cudaMalloc((void**)&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc((void**)&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc((void**)&deviceOutput, inputLength * sizeof(DataType));

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Start timer for memory copy and computation
    start_timer();
    
    int threadsPerBlock = 256;
    // Process data in segments using streams
    for (int i = 0; i < inputLength; i += segmentSize) {
        int currentSize = min(segmentSize, inputLength - i);
        int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        int streamIdx = (i / segmentSize) % NUM_STREAMS;

        // Asynchronous memory copy to device
        cudaMemcpyAsync(deviceInput1 + i, hostInput1 + i,
                       currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                       streams[streamIdx]);
        cudaMemcpyAsync(deviceInput2 + i, hostInput2 + i,
                       currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                       streams[streamIdx]);

        // Launch kernel
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>
            (deviceInput1 + i, deviceInput2 + i, deviceOutput + i, currentSize);

        // Asynchronous memory copy back to host
        cudaMemcpyAsync(hostOutput + i, deviceOutput + i,
                       currentSize * sizeof(DataType), cudaMemcpyDeviceToHost,
                       streams[streamIdx]);
    }

    // Synchronize all streams
    cudaDeviceSynchronize();
    stop_timer("Total execution time with streams");

    // Verify results - same as lab2
    bool match = true;
    for (int i = 0; i < inputLength; i++) {
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, hostOutput[i], resultRef[i]);
            match = false;
            break;
        }
    }
    printf(match ? "Results match.\n" : "Results do not match.\n");

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
