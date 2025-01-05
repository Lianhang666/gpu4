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
    int inputLength;
    DataType *hostInput1, *hostInput2, *hostOutput, *resultRef;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;

    if (argc > 1) {
        inputLength = atoi(argv[1]);
    } else {
        printf("Please provide the input length as an argument.\n");
        return 1;
    }
    printf("The input length is %d\n", inputLength);

    // 使用页锁定内存以支持异步传输
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

    // 计算每个流处理的数据大小
    int segmentSize = inputLength / NUM_STREAMS;
    int threadsPerBlock = 256;
    int blocksPerSegment = (segmentSize + threadsPerBlock - 1) / threadsPerBlock;

    start_timer();
    
    // 流水线执行
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * segmentSize;
        // 当前流处理的实际大小（处理最后一个段可能不足segmentSize）
        int currentSize = (i == NUM_STREAMS - 1) ? 
            inputLength - offset : segmentSize;

        // 使用当前流异步传输输入数据
        cudaMemcpyAsync(deviceInput1 + offset, 
                       hostInput1 + offset,
                       currentSize * sizeof(DataType), 
                       cudaMemcpyHostToDevice, 
                       streams[i]);
        cudaMemcpyAsync(deviceInput2 + offset, 
                       hostInput2 + offset,
                       currentSize * sizeof(DataType), 
                       cudaMemcpyHostToDevice, 
                       streams[i]);

        // 在当前流上启动核函数
        vecAdd<<<blocksPerSegment, threadsPerBlock, 0, streams[i]>>>
            (deviceInput1 + offset, 
             deviceInput2 + offset, 
             deviceOutput + offset, 
             currentSize);

        // 使用当前流异步传输结果
        cudaMemcpyAsync(hostOutput + offset, 
                       deviceOutput + offset,
                       currentSize * sizeof(DataType), 
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }

    // 等待所有流完成
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
    cudaFreeHost(hostInput1);    // 释放页锁定内存
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    free(resultRef);
    
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
