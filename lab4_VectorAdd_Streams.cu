#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 定时器结构和数据类型定义
struct timeval start, stop;
#define DataType double
#define NUM_STREAMS 4

// 向量加法核函数
__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

// 定时器函数
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
    
    // 从命令行参数获取输入长度和段大小
    if (argc > 2) {
        inputLength = atoi(argv[1]);
        segmentSize = atoi(argv[2]);
    } else {
        printf("Usage: %s <input_length> <segment_size>\n", argv[0]);
        return 1;
    }
    printf("Input length: %d, Segment size: %d\n", inputLength, segmentSize);

    // 分配页锁定的主机内存（用于异步传输）
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));
    resultRef = (DataType*)malloc(inputLength * sizeof(DataType));

    // 初始化输入数据
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 100;
        hostInput2[i] = rand() % 100;
        resultRef[i] = hostInput1[i] + hostInput2[i];  // 计算CPU参考结果
    }

    // 分配设备内存
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    // 创建CUDA流
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 开始计时
    start_timer();

    // 使用多个流处理数据段
    int threadsPerBlock = 256;
    
    // 对每个大段进行处理
    for (int offset = 0; offset < inputLength; offset += segmentSize * NUM_STREAMS) {
        // 每个流处理一个小段
        for (int i = 0; i < NUM_STREAMS && (offset + i * segmentSize) < inputLength; i++) {
            int currentOffset = offset + i * segmentSize;
            int currentSize = min(segmentSize, inputLength - currentOffset);
            int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

            // 异步数据传输到GPU
            cudaMemcpyAsync(deviceInput1 + currentOffset, hostInput1 + currentOffset,
                           currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                           streams[i]);
            cudaMemcpyAsync(deviceInput2 + currentOffset, hostInput2 + currentOffset,
                           currentSize * sizeof(DataType), cudaMemcpyHostToDevice,
                           streams[i]);
            
            // 在对应流上启动核函数
            vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
                (deviceInput1 + currentOffset, deviceInput2 + currentOffset, 
                 deviceOutput + currentOffset, currentSize);

            // 异步数据传输回主机
            cudaMemcpyAsync(hostOutput + currentOffset, deviceOutput + currentOffset,
                           currentSize * sizeof(DataType), cudaMemcpyDeviceToHost,
                           streams[i]);
        }
    }
    
    // 同步所有流
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // 停止计时
    stop_timer("Total execution time with streams");

    // 验证结果
    bool match = true;
    for (int i = 0; i < inputLength; i++) {
        if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
            match = false;
            printf("Mismatch at position %d: GPU = %f, CPU = %f\n",
                   i, hostOutput[i], resultRef[i]);
            break;
        }
    }
    printf("Result verification: %s\n", match ? "PASSED" : "FAILED");

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
