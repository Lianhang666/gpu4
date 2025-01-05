#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval start, stop;
#define DataType double
#define NUM_STREAMS 4

// Timer functions
void start_timer() {
    gettimeofday(&start, NULL);
}

void stop_timer(const char* message) {
    gettimeofday(&stop, NULL);
    double elapsedTime = (stop.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (stop.tv_usec - start.tv_usec) / 1000.0;
    printf("%s: %.2f ms\n", message, elapsedTime);
}

// Pin memory pool structure
typedef struct {
    DataType* input1;
    DataType* input2;
    DataType* output;
    size_t size;
} PinnedMemoryPool;

// 初始化pin memory池
void initPinnedMemoryPool(PinnedMemoryPool* pool, size_t size) {
    pool->size = size;
    cudaMallocHost(&pool->input1, size * sizeof(DataType));
    cudaMallocHost(&pool->input2, size * sizeof(DataType));
    cudaMallocHost(&pool->output, size * sizeof(DataType));
}

// 释放pin memory池
void freePinnedMemoryPool(PinnedMemoryPool* pool) {
    cudaFreeHost(pool->input1);
    cudaFreeHost(pool->input2);
    cudaFreeHost(pool->output);
}

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char **argv) {
    int inputLength, segmentSize;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    DataType *resultRef;

    // 参数检查
    if (argc > 2) {
        inputLength = atoi(argv[1]);
        segmentSize = atoi(argv[2]);
    } else {
        printf("Usage: %s <input_length> <segment_size>\n", argv[0]);
        return 1;
    }

    // 对齐段大小
    segmentSize = (segmentSize + 255) & ~255;
    printf("Input length: %d, Adjusted segment size: %d\n", inputLength, segmentSize);

    // 初始化pin memory池
    PinnedMemoryPool memPool;
    initPinnedMemoryPool(&memPool, inputLength);
    
    // 分配参考内存（不需要pin）
    resultRef = (DataType*)malloc(inputLength * sizeof(DataType));

    // 初始化数据
    for (int i = 0; i < inputLength; i++) {
        memPool.input1[i] = rand() % 100;
        memPool.input2[i] = rand() % 100;
        resultRef[i] = memPool.input1[i] + memPool.input2[i];
    }

    // 分配GPU内存
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    // 创建CUDA流和事件
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    int threadsPerBlock = 256;
    start_timer();

    // 划分数据并处理
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * (inputLength / NUM_STREAMS);
        int currentSize = (i == NUM_STREAMS - 1) ? 
            (inputLength - offset) : (inputLength / NUM_STREAMS);

        // H2D阶段
        cudaMemcpyAsync(deviceInput1 + offset, 
                       memPool.input1 + offset,
                       currentSize * sizeof(DataType),
                       cudaMemcpyHostToDevice, 
                       streams[i]);
        cudaMemcpyAsync(deviceInput2 + offset, 
                       memPool.input2 + offset,
                       currentSize * sizeof(DataType),
                       cudaMemcpyHostToDevice, 
                       streams[i]);

        // Kernel阶段
        int blocksPerGrid = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
            (deviceInput1 + offset, deviceInput2 + offset,
             deviceOutput + offset, currentSize);

        // D2H阶段
        cudaMemcpyAsync(memPool.output + offset, 
                       deviceOutput + offset,
                       currentSize * sizeof(DataType),
                       cudaMemcpyDeviceToHost, 
                       streams[i]);

        // 记录事件
        cudaEventRecord(events[i], streams[i]);
    }

    // 等待所有事件完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaEventSynchronize(events[i]);
    }

    stop_timer("Total execution time with streams");

    // 验证结果
    bool match = true;
    for (int i = 0; i < inputLength; i++) {
        if (fabs(memPool.output[i] - resultRef[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, memPool.output[i], resultRef[i]);
            match = false;
            break;
        }
    }
    printf(match ? "Results match.\n" : "Results do not match.\n");

    // 清理资源
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaEventDestroy(events[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    freePinnedMemoryPool(&memPool);
    free(resultRef);

    return 0;
}
