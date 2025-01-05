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

typedef struct {
    DataType* input1;
    DataType* input2;
    DataType* output;
} MemoryPool;

void initMemoryPool(MemoryPool* pool, int vectorLength) {
    cudaMallocHost(&pool->input1, vectorLength * sizeof(DataType));
    cudaMallocHost(&pool->input2, vectorLength * sizeof(DataType));
    cudaMallocHost(&pool->output, vectorLength * sizeof(DataType));
}

void freeMemoryPool(MemoryPool* pool) {
    cudaFreeHost(pool->input1);
    cudaFreeHost(pool->input2);
    cudaFreeHost(pool->output);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <vector_length>\n", argv[0]);
        return 1;
    }

    int vectorLength = atoi(argv[1]);
    int segmentSize = vectorLength / NUM_STREAMS;
    
    printf("Vector Length: %d, Segment Size: %d\n", vectorLength, segmentSize);

    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    DataType *resultRef;
    
    // 初始化内存池
    MemoryPool pool;
    initMemoryPool(&pool, vectorLength);
    resultRef = (DataType*)malloc(vectorLength * sizeof(DataType));

    // 初始化数据
    for (int i = 0; i < vectorLength; i++) {
        pool.input1[i] = rand() % 100;
        pool.input2[i] = rand() % 100;
        resultRef[i] = pool.input1[i] + pool.input2[i];
    }

    // 分配GPU内存
    cudaMalloc(&deviceInput1, vectorLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, vectorLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, vectorLength * sizeof(DataType));

    // 创建CUDA流
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threadsPerBlock = 256;
    int blocksPerSegment = (segmentSize + threadsPerBlock - 1) / threadsPerBlock;

    start_timer();

    // Stage 1: 批量H2D传输
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * segmentSize;
        
        cudaMemcpyAsync(deviceInput1 + offset, 
                       pool.input1 + offset,
                       segmentSize * sizeof(DataType),
                       cudaMemcpyHostToDevice, 
                       streams[i]);
        cudaMemcpyAsync(deviceInput2 + offset, 
                       pool.input2 + offset,
                       segmentSize * sizeof(DataType),
                       cudaMemcpyHostToDevice, 
                       streams[i]);
    }

    // Stage 2: 批量kernel执行
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * segmentSize;
        
        vecAdd<<<blocksPerSegment, threadsPerBlock, 0, streams[i]>>>
            (deviceInput1 + offset, deviceInput2 + offset,
             deviceOutput + offset, segmentSize);
    }

    // Stage 3: 批量D2H传输
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * segmentSize;
        
        cudaMemcpyAsync(pool.output + offset, 
                       deviceOutput + offset,
                       segmentSize * sizeof(DataType),
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
    }

    // 最后统一同步
    cudaDeviceSynchronize();
    stop_timer("Total execution time with streams");

    // 验证结果
    bool match = true;
    for (int i = 0; i < vectorLength; i++) {
        if (fabs(pool.output[i] - resultRef[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, pool.output[i], resultRef[i]);
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
    freeMemoryPool(&pool);
    free(resultRef);

    return 0;
}
