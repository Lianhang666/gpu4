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

    // 参数检查
    if (argc > 2) {
        inputLength = atoi(argv[1]);
        segmentSize = atoi(argv[2]);
    } else {
        printf("Usage: %s <input_length> <segment_size>\n", argv[0]);
        return 1;
    }
    
    // 确保段大小是256的倍数（优化内存访问）
    segmentSize = (segmentSize + 255) & ~255;
    printf("Input length: %d, Adjusted segment size: %d\n", inputLength, segmentSize);

    // 分配页锁定内存 - 移到开始以减少计时影响
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));
    resultRef = (DataType*)malloc(inputLength * sizeof(DataType));

    // 初始化输入数据
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

    int threadsPerBlock = 256;
    start_timer();

    // Stage 1: 批量H2D传输
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * (inputLength / NUM_STREAMS);
        int currentSize = (i == NUM_STREAMS - 1) ? 
            (inputLength - offset) : (inputLength / NUM_STREAMS);

        cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset,
                       currentSize * sizeof(DataType),
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset,
                       currentSize * sizeof(DataType),
                       cudaMemcpyHostToDevice, streams[i]);
    }

    // Stage 2: 批量kernel执行
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * (inputLength / NUM_STREAMS);
        int currentSize = (i == NUM_STREAMS - 1) ? 
            (inputLength - offset) : (inputLength / NUM_STREAMS);
        int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
            (deviceInput1 + offset, deviceInput2 + offset,
             deviceOutput + offset, currentSize);
    }

    // Stage 3: 批量D2H传输
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * (inputLength / NUM_STREAMS);
        int currentSize = (i == NUM_STREAMS - 1) ? 
            (inputLength - offset) : (inputLength / NUM_STREAMS);

        cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset,
                       currentSize * sizeof(DataType),
                       cudaMemcpyDeviceToHost, streams[i]);
    }

    // 一次性同步所有流
    cudaDeviceSynchronize();
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
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    free(resultRef);

    return 0;
}
