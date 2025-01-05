#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

// 让 kernel 做更多运算，增加执行时间
#define FLAOT_OPS 100

inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 简单的计时函数（CPU端）
double wallTime() {
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

// 在 GPU 上测量时间可以用 cudaEvent
float gpuTime(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop); // ms
    return ms;
}

// 让 kernel 做比“单次加法”更多的计算，延长执行时间
__global__ void vecAddKernel(const float *in1, const float *in2, float *out, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float tmp = in1[idx] + in2[idx];
        // 做更多运算来增加 kernel 的执行时间
        for(int k = 0; k < FLAOT_OPS; k++){
            tmp = tmp * 1.000001f + 0.0001f;
        }
        out[idx] = tmp;
    }
}

// --------------------------
// 单流版本：一次性把所有数据拷贝上去 -> 核函数 -> 拷回
// 返回 GPU 端所用时间(ms)和 总 wall-clock 时间(s)
// --------------------------
void testSingleStream(const float* h_in1, const float* h_in2, float* h_out,
                      int len, float &gpuTimeMs, double &wallTimeSec)
{
    // 分配 GPU
    float *d_in1, *d_in2, *d_out;
    checkCudaError(cudaMalloc(&d_in1, len * sizeof(float)), "Malloc d_in1");
    checkCudaError(cudaMalloc(&d_in2, len * sizeof(float)), "Malloc d_in2");
    checkCudaError(cudaMalloc(&d_out, len * sizeof(float)), "Malloc d_out");

    double t0 = wallTime();

    // 使用 cudaEvent 测量 GPU 时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录起始事件
    cudaEventRecord(start, 0);

    // 拷贝到 GPU
    checkCudaError(cudaMemcpy(d_in1, h_in1, len*sizeof(float), cudaMemcpyHostToDevice), "Memcpy H2D in1");
    checkCudaError(cudaMemcpy(d_in2, h_in2, len*sizeof(float), cudaMemcpyHostToDevice), "Memcpy H2D in2");

    // 启动 kernel
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    vecAddKernel<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, len);

    // 拷贝结果回主机
    checkCudaError(cudaMemcpy(h_out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost), "Memcpy D2H out");

    // 记录停止事件
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算 GPU 端总时间
    gpuTimeMs = gpuTime(start, stop);

    double t1 = wallTime();
    wallTimeSec = t1 - t0;

    // 清理
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// --------------------------
// 多流版本：把大向量按 segmentSize 分多批次 + NUM_STREAMS 并行做异步复制 + kernel + 复制回
// 返回 GPU 端所用时间(ms)和 总 wall-clock 时间(s)
// --------------------------
void testMultiStream(const float* h_in1, const float* h_in2, float* h_out,
                     int len, int segmentSize, int numStreams,
                     float &gpuTimeMs, double &wallTimeSec)
{
    // 分配 GPU (为 simplicity, 一次性分配 len 大小)
    float *d_in1, *d_in2, *d_out;
    checkCudaError(cudaMalloc(&d_in1, len * sizeof(float)), "Malloc d_in1");
    checkCudaError(cudaMalloc(&d_in2, len * sizeof(float)), "Malloc d_in2");
    checkCudaError(cudaMalloc(&d_out, len * sizeof(float)), "Malloc d_out");

    double t0 = wallTime();

    // 用 cudaEvent 在 GPU 端测整体时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 创建 streams
    cudaStream_t *streams = (cudaStream_t*)malloc(numStreams * sizeof(cudaStream_t));
    for(int i=0; i<numStreams; i++){
        cudaStreamCreate(&streams[i]);
    }

    // 记录起始事件
    cudaEventRecord(start, 0);

    // 分批处理
    const int blockSize = 256;
    for(int base = 0; base < len; base += segmentSize * numStreams){
        for(int s = 0; s < numStreams; s++){
            int offset = base + s * segmentSize;
            if(offset >= len) break;  // 已超范围
            int size = (offset + segmentSize <= len) ? segmentSize : (len - offset);

            // 异步拷贝 H->D
            cudaMemcpyAsync(d_in1 + offset, h_in1 + offset, size*sizeof(float),
                            cudaMemcpyHostToDevice, streams[s]);
            cudaMemcpyAsync(d_in2 + offset, h_in2 + offset, size*sizeof(float),
                            cudaMemcpyHostToDevice, streams[s]);

            // 启动 kernel
            int gridSize = (size + blockSize - 1) / blockSize;
            vecAddKernel<<<gridSize, blockSize, 0, streams[s]>>>
                (d_in1 + offset, d_in2 + offset, d_out + offset, size);

            // 异步拷贝 D->H
            cudaMemcpyAsync(h_out + offset, d_out + offset, size*sizeof(float),
                            cudaMemcpyDeviceToHost, streams[s]);
        }
    }

    // 同步所有流
    for(int s=0; s<numStreams; s++){
        cudaStreamSynchronize(streams[s]);
    }

    // 记录停止事件
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算 GPU 端总时间
    gpuTimeMs = gpuTime(start, stop);

    double t1 = wallTime();
    wallTimeSec = t1 - t0;

    // 清理
    for(int s=0; s<numStreams; s++){
        cudaStreamDestroy(streams[s]);
    }
    free(streams);

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// --------------------------
// 验证结果 correctness
// --------------------------
bool verify(const float* h_ref, const float* h_out, int len) {
    for(int i=0; i<len; i++){
        if(fabs(h_ref[i] - h_out[i]) > 1e-2){ 
            // 因为我们做了很多浮点操作,误差阈值可稍大
            // 也可用 1e-3 视情况而定
            return false;
        }
    }
    return true;
}

int main() {
    // 准备一些不同向量长度
    // 你可以根据需要修改这里
    int vectorLengths[] = {1'000'000, 2'000'000, 5'000'000, 10'000'000};
    int numLengths = sizeof(vectorLengths)/sizeof(int);

    // 准备一些 segmentSizes
    int segmentSizes[] = {1000, 5000, 10000, 50000, 100000};
    int numSegments = sizeof(segmentSizes)/sizeof(int);

    // 用多少个 stream
    const int NUM_STREAMS = 4;

    // 重复执行多少次取平均 (排除抖动)
    const int REPEAT = 3;

    printf("FLAOT_OPS = %d (extra ops in kernel)\n", FLAOT_OPS);
    printf("NUM_STREAMS = %d\n\n", NUM_STREAMS);

    // 打印 CSV 表头：向量长度,单流耗时(平均GPU ms),单流总时间(s),
    //               多流(segmentSize=xxx)平均GPU ms, 多流总时间(s), ...
    // 这部分可自行调整，根据你想要的格式
    printf("VectorLength");
    printf(",SingleStream_GPUms,SingleStream_WallSec");
    for(int segId=0; segId<numSegments; segId++){
        printf(",MultiStream(seg=%d)_GPUms,MultiStream(seg=%d)_WallSec", 
               segmentSizes[segId], segmentSizes[segId]);
    }
    printf("\n");

    // 循环测试
    for(int iLen=0; iLen<numLengths; iLen++){
        int len = vectorLengths[iLen];

        // 分配 Host 内存
        float* h_in1 = (float*)malloc(len * sizeof(float));
        float* h_in2 = (float*)malloc(len * sizeof(float));
        float* h_out_single = (float*)malloc(len * sizeof(float));
        float* h_out_multi  = (float*)malloc(len * sizeof(float));
        float* h_ref        = (float*)malloc(len * sizeof(float));

        // 初始化数据
        srand(1234);
        for(int i=0; i<len; i++){
            float a = (float)(rand() % 100);
            float b = (float)(rand() % 100);
            h_in1[i] = a;
            h_in2[i] = b;
            // 这里先做简单加法当作 reference，再加点额外运算保持和 kernel 同样的逻辑
            float tmp = a + b;
            for(int k=0; k<FLAOT_OPS; k++){
                tmp = tmp * 1.000001f + 0.0001f;
            }
            h_ref[i] = tmp;
        }

        // 先测试单流 - 多次执行取平均
        float sumGpuMs_single = 0.0f;
        double sumWall_single = 0.0;
        for(int r=0; r<REPEAT; r++){
            float gms;
            double wsec;
            testSingleStream(h_in1, h_in2, h_out_single, len, gms, wsec);

            sumGpuMs_single += gms;
            sumWall_single  += wsec;
        }
        float avgGpuMs_single = sumGpuMs_single / REPEAT;
        double avgWall_single = sumWall_single / REPEAT;

        // 验证结果
        if(!verify(h_ref, h_out_single, len)){
            fprintf(stderr, "ERROR: SingleStream results do not match!\n");
        }

        // 打印前半部分（向量长度, 单流统计）
        printf("%d,%.3f,%.3f", len, avgGpuMs_single, avgWall_single);

        // 再测试多流 - 不同 segmentSizes
        for(int segId=0; segId<numSegments; segId++){
            int segSize = segmentSizes[segId];
            float sumGpuMs_multi = 0.0f;
            double sumWall_multi = 0.0;
            for(int r=0; r<REPEAT; r++){
                float gms;
                double wsec;
                testMultiStream(h_in1, h_in2, h_out_multi, len, segSize, NUM_STREAMS, gms, wsec);

                sumGpuMs_multi += gms;
                sumWall_multi  += wsec;
            }
            float avgGpuMs_multi = sumGpuMs_multi / REPEAT;
            double avgWall_multi = sumWall_multi / REPEAT;

            // 验证结果
            if(!verify(h_ref, h_out_multi, len)){
                fprintf(stderr, "ERROR: MultiStream(seg=%d) results do not match!\n", segSize);
            }

            // 打印 CSV: GPU时间,Wall-clock时间
            printf(",%.3f,%.3f", avgGpuMs_multi, avgWall_multi);
        }
        printf("\n");

        free(h_in1);
        free(h_in2);
        free(h_out_single);
        free(h_out_multi);
        free(h_ref);
    }

    return 0;
}
