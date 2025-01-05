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
    DataType *hostInput1, *hostInput2, *hostOutput;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    cudaStream_t streams[NUM_STREAMS];
    
    // Get input length and segment size from command line
    if (argc > 2) {
        inputLength = atoi(argv[1]);
        segmentSize = atoi(argv[2]);
    } else {
        printf("Usage: %s <input_length> <segment_size>\n", argv[0]);
        return 1;
    }
    printf("Input length: %d, Segment size: %d\n", inputLength, segmentSize);

    // 分配页锁定内存
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));

    // Initialize input arrays
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType)RAND_MAX;
        hostInput2[i] = rand() / (DataType)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    // Create CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Start timing
    start_timer();

    // Process data in segments using multiple streams
    int threadsPerBlock = 256;
    for (int offset = 0; offset < inputLength; offset += segmentSize * NUM_STREAMS) {
        for (int i = 0; i < NUM_STREAMS && (offset + i * segmentSize) < inputLength; i++) {
            int currentOffset = offset + i * segmentSize;
            int currentSize = min(segmentSize, inputLength - currentOffset);
            int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

            // Asynchronous memory transfers and kernel execution
            cudaMemcpyAsync(deviceInput1 + currentOffset, hostInput1 + currentOffset,
                           currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                           streams[i]);
            cudaMemcpyAsync(deviceInput2 + currentOffset, hostInput2 + currentOffset,
                           currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                           streams[i]);

            vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
                (deviceInput1 + currentOffset, deviceInput2 + currentOffset,
                 deviceOutput + currentOffset, currentSize);

            cudaMemcpyAsync(hostOutput + currentOffset, deviceOutput + currentOffset,
                           currentSize * sizeof(DataType), cudaMemcpyDeviceToHost,
                           streams[i]);
        }
    }

    // Synchronize all streams before stopping timer
    cudaDeviceSynchronize();
    stop_timer("Total execution time with streams");

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);

    return 0;
}
