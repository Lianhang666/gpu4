#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define FLAOT_OPS 100

inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

double wallTime() {
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

float gpuTime(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop); // ms
    return ms;
}

// A kernel that does additional operations to make it run longer than a simple addition
__global__ void vecAddKernel(const float *in1, const float *in2, float *out, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float tmp = in1[idx] + in2[idx];
        for(int k = 0; k < FLAOT_OPS; k++){
            tmp = tmp * 1.000001f + 0.0001f;
        }
        out[idx] = tmp;
    }
}

// Single-stream version: copy everything to GPU, run kernel, copy results back
void testSingleStream(const float* h_in1, const float* h_in2, float* h_out,
                      int len, float &gpuTimeMs, double &wallTimeSec)
{
    float *d_in1, *d_in2, *d_out;
    checkCudaError(cudaMalloc(&d_in1, len * sizeof(float)), "Malloc d_in1");
    checkCudaError(cudaMalloc(&d_in2, len * sizeof(float)), "Malloc d_in2");
    checkCudaError(cudaMalloc(&d_out, len * sizeof(float)), "Malloc d_out");

    double t0 = wallTime();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    checkCudaError(cudaMemcpy(d_in1, h_in1, len*sizeof(float), cudaMemcpyHostToDevice), "Memcpy H2D in1");
    checkCudaError(cudaMemcpy(d_in2, h_in2, len*sizeof(float), cudaMemcpyHostToDevice), "Memcpy H2D in2");

    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    vecAddKernel<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, len);

    checkCudaError(cudaMemcpy(h_out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost), "Memcpy D2H out");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    gpuTimeMs = gpuTime(start, stop);

    double t1 = wallTime();
    wallTimeSec = t1 - t0;

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Multi-stream version: split the large vector into segments and use NUM_STREAMS
void testMultiStream(const float* h_in1, const float* h_in2, float* h_out,
                     int len, int segmentSize, int numStreams,
                     float &gpuTimeMs, double &wallTimeSec)
{
    float *d_in1, *d_in2, *d_out;
    checkCudaError(cudaMalloc(&d_in1, len * sizeof(float)), "Malloc d_in1");
    checkCudaError(cudaMalloc(&d_in2, len * sizeof(float)), "Malloc d_in2");
    checkCudaError(cudaMalloc(&d_out, len * sizeof(float)), "Malloc d_out");

    double t0 = wallTime();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStream_t *streams = (cudaStream_t*)malloc(numStreams * sizeof(cudaStream_t));
    for(int i = 0; i < numStreams; i++){
        cudaStreamCreate(&streams[i]);
    }

    const int blockSize = 256;
    for(int base = 0; base < len; base += segmentSize * numStreams){
        for(int s = 0; s < numStreams; s++){
            int offset = base + s * segmentSize;
            if(offset >= len) break;
            int size = (offset + segmentSize <= len) ? segmentSize : (len - offset);

            cudaMemcpyAsync(d_in1 + offset, h_in1 + offset,
                            size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
            cudaMemcpyAsync(d_in2 + offset, h_in2 + offset,
                            size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);

            int gridSize = (size + blockSize - 1) / blockSize;
            vecAddKernel<<<gridSize, blockSize, 0, streams[s]>>>
                (d_in1 + offset, d_in2 + offset, d_out + offset, size);

            cudaMemcpyAsync(h_out + offset, d_out + offset,
                            size*sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
        }
    }

    for(int s = 0; s < numStreams; s++){
        cudaStreamSynchronize(streams[s]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    gpuTimeMs = gpuTime(start, stop);

    double t1 = wallTime();
    wallTimeSec = t1 - t0;

    for(int s = 0; s < numStreams; s++){
        cudaStreamDestroy(streams[s]);
    }
    free(streams);

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

bool verify(const float* h_ref, const float* h_out, int len) {
    for(int i = 0; i < len; i++){
        if(fabs(h_ref[i] - h_out[i]) > 1e-2){
            return false;
        }
    }
    return true;
}

int main() {
    int vectorLengths[] = {1'000'000, 2'000'000, 5'000'000, 10'000'000};
    int numLengths = sizeof(vectorLengths)/sizeof(int);

    int segmentSizes[] = {1000, 5000, 10000, 50000, 100000};
    int numSegments = sizeof(segmentSizes)/sizeof(int);

    const int NUM_STREAMS = 4;
    const int REPEAT = 3;

    printf("FLAOT_OPS = %d\n", FLAOT_OPS);
    printf("NUM_STREAMS = %d\n\n", NUM_STREAMS);

    printf("VectorLength");
    printf(",SingleStream_GPUms,SingleStream_WallSec");
    for(int segId = 0; segId < numSegments; segId++){
        printf(",MultiStream(seg=%d)_GPUms,MultiStream(seg=%d)_WallSec",
               segmentSizes[segId], segmentSizes[segId]);
    }
    printf("\n");

    for(int iLen = 0; iLen < numLengths; iLen++){
        int len = vectorLengths[iLen];

        float* h_in1 = (float*)malloc(len * sizeof(float));
        float* h_in2 = (float*)malloc(len * sizeof(float));
        float* h_out_single = (float*)malloc(len * sizeof(float));
        float* h_out_multi  = (float*)malloc(len * sizeof(float));
        float* h_ref        = (float*)malloc(len * sizeof(float));

        srand(1234);
        for(int i = 0; i < len; i++){
            float a = (float)(rand() % 100);
            float b = (float)(rand() % 100);
            h_in1[i] = a;
            h_in2[i] = b;
            float tmp = a + b;
            for(int k = 0; k < FLAOT_OPS; k++){
                tmp = tmp * 1.000001f + 0.0001f;
            }
            h_ref[i] = tmp;
        }

        float sumGpuMs_single = 0.0f;
        double sumWall_single = 0.0;
        for(int r = 0; r < REPEAT; r++){
            float gms;
            double wsec;
            testSingleStream(h_in1, h_in2, h_out_single, len, gms, wsec);
            sumGpuMs_single += gms;
            sumWall_single  += wsec;
        }
        float avgGpuMs_single = sumGpuMs_single / REPEAT;
        double avgWall_single = sumWall_single / REPEAT;

        if(!verify(h_ref, h_out_single, len)){
            fprintf(stderr, "ERROR: SingleStream results do not match!\n");
        }

        printf("%d,%.3f,%.3f", len, avgGpuMs_single, avgWall_single);

        for(int segId = 0; segId < numSegments; segId++){
            int segSize = segmentSizes[segId];
            float sumGpuMs_multi = 0.0f;
            double sumWall_multi = 0.0;
            for(int r = 0; r < REPEAT; r++){
                float gms;
                double wsec;
                testMultiStream(h_in1, h_in2, h_out_multi, len, segSize, NUM_STREAMS, gms, wsec);
                sumGpuMs_multi += gms;
                sumWall_multi  += wsec;
            }
            float avgGpuMs_multi = sumGpuMs_multi / REPEAT;
            double avgWall_multi = sumWall_multi / REPEAT;

            if(!verify(h_ref, h_out_multi, len)){
                fprintf(stderr, "ERROR: MultiStream(seg=%d) results do not match!\n", segSize);
            }

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
