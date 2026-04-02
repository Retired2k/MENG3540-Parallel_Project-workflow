#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cv;
using namespace std;

#define BLOCK_DIM 256

// =====================================================
// CUDA Error Check
// =====================================================
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        cerr << "CUDA Error: " << cudaGetErrorString(err)                    \
             << " at line " << __LINE__ << endl;                             \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

// =====================================================
// Kernel 1: Extract Green Channel
// OpenCV uses BGR, so green is index +1
// =====================================================
__global__ void extractGreenKernel(const unsigned char* input,
                                   float* green,
                                   int width,
                                   int height,
                                   int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIdx = y * width + x;
        int imgIdx = pixelIdx * channels;
        green[pixelIdx] = static_cast<float>(input[imgIdx + 1]);
    }
}

// =====================================================
// Kernel 2: Shared Memory Reduction Kernel
// Each thread loads 2 elements and reduces in shared memory
// =====================================================
__global__ void SharedMemorySumReductionKernel(float* input, float* output, int n)
{
    __shared__ float input_s[BLOCK_DIM];

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val1 = 0.0f;
    float val2 = 0.0f;

    if (i < n)
        val1 = input[i];

    if (i + blockDim.x < n)
        val2 = input[i + blockDim.x];

    input_s[t] = val1 + val2;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();

        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }

        if (stride == 1)
            break;
    }

    if (t == 0) {
        output[blockIdx.x] = input_s[0];
    }
}

// =====================================================
// Host Reduction Function
// =====================================================
float reduceOnGPU(float* d_input, int n)
{
    int currentSize = n;
    float* d_in = d_input;
    float* d_out = nullptr;

    while (currentSize > 1) {
        int blocks = (currentSize + (BLOCK_DIM * 2 - 1)) / (BLOCK_DIM * 2);

        CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));

        SharedMemorySumReductionKernel<<<blocks, BLOCK_DIM>>>(d_in, d_out, currentSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_in != d_input) {
            CUDA_CHECK(cudaFree(d_in));
        }

        d_in = d_out;
        currentSize = blocks;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost));

    if (d_in != d_input) {
        CUDA_CHECK(cudaFree(d_in));
    }

    return result;
}

// =====================================================
// Main
// Capture one image from webcam, then process once
// =====================================================
int main()
{
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    Mat frame;

    cap >> frame;

    if (frame.empty()) {
        cerr << "Error: Could not capture frame." << endl;
        return -1;
    }

    int width = frame.cols;
    int height = frame.rows;
    int channels = frame.channels();
    int numPixels = width * height;
    int imageSize = width * height * channels * sizeof(unsigned char);

    unsigned char* d_frame = nullptr;
    float* d_green = nullptr;

    CUDA_CHECK(cudaMalloc(&d_frame, imageSize));
    CUDA_CHECK(cudaMalloc(&d_green, numPixels * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_frame, frame.data, imageSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    float kernelTime = 0.0f;
    float reductionTime = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    extractGreenKernel<<<gridSize, blockSize>>>(d_frame, d_green, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    float totalGreen = reduceOnGPU(d_green, numPixels);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&reductionTime, start, stop));

    float avgGreen = totalGreen / numPixels;
    float greenPercent = (avgGreen / 255.0f) * 100.0f;

    cout << "Kernel Execution Time: " << kernelTime << " ms" << endl;
    cout << "Reduction Time: " << reductionTime << " ms" << endl;

    cout << "\n=== RESULTS ===" << endl;
    cout << "Total Green Sum: " << totalGreen << endl;
    cout << "Average Green: " << avgGreen << endl;
    cout << "Green %: " << greenPercent << "%" << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_frame));
    CUDA_CHECK(cudaFree(d_green));

    cap.release();
    return 0;
}
