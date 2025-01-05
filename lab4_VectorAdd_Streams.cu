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

    // 分配页锁定内存
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));
    resultRef = (DataType*)malloc(inputLength * sizeof(DataType));  // 参考结果可以使用普通内存

    // 初始化输入数组
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 100;
        hostInput2[i] = rand() % 100;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    // 分配GPU内存
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    // 创建CUDA流
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 计算每个流处理的段大小
    int segmentsPerStream = (inputLength + segmentSize - 1) / segmentSize;
    int threadsPerBlock = 256;

    start_timer();

    // Stage 1: 所有Host to Device传输
    for (int i = 0; i < segmentsPerStream; i++) {
        int offset = i * segmentSize;
        int currentSize = min(segmentSize, inputLength - offset);
        int streamIdx = i % NUM_STREAMS;

        cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset,
                       currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                       streams[streamIdx]);
        cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset,
                       currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                       streams[streamIdx]);
    }

    // Stage 2: 所有kernel执行
    for (int i = 0; i < segmentsPerStream; i++) {
        int offset = i * segmentSize;
        int currentSize = min(segmentSize, inputLength - offset);
        int streamIdx = i % NUM_STREAMS;
        int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>
            (deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, currentSize);
    }

    // Stage 3: 所有Device to Host传输
    for (int i = 0; i < segmentsPerStream; i++) {
        int offset = i * segmentSize;
        int currentSize = min(segmentSize, inputLength - offset);
        int streamIdx = i % NUM_STREAMS;

        cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset,
                       currentSize * sizeof(DataType), cudaMemcpyDeviceToHost,
                       streams[streamIdx]);
    }

    // 同步所有流
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    stop_timer("Total execution time with streams");

    // 验证结果
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

    // 清理资源
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // 释放GPU内存
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // 释放CPU内存
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    free(resultRef);

    return 0;
}
