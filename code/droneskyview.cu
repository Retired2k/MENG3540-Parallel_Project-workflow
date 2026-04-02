#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cv;
using namespace std;

// =====================================================
// Error checking
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
// Kernel 1: extract green channel from BGR image
// OpenCV uses BGR, so green is index +1
// =====================================================
__global__ void ExtractGreenKernel(unsigned char* image, float* green, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int imageIdx = pixelIdx * 3;

    green[pixelIdx] = (float)image[imageIdx + 1];
}

// =====================================================
// Kernel 2: block reduction
// Each block reduces 2 * blockDim.x values
// =====================================================
__global__ void ReductionKernel(float* input, float* output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;

    if (i < n)
        sum = input[i];

    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =====================================================
// Host function to finish reduction
// =====================================================
float ReduceSum(float* d_input, int n)
{
    int threads = 256;
    int currentSize = n;

    float* d_in = d_input;
    float* d_out = nullptr;

    while (currentSize > 1) {
        int blocks = (currentSize + (threads * 2 - 1)) / (threads * 2);

        CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));

        ReductionKernel << <blocks, threads, threads * sizeof(float) >> > (d_in, d_out, currentSize);
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
// =====================================================
int main()
{
    VideoCapture cap(0);   // open webcam
    if (!cap.isOpened())
    {
        cout << "Camera open failed\n";
        return -1;
    }

    Mat frame;

    unsigned char* d_image = nullptr;
    float* d_green = nullptr;

    int allocatedWidth = 0;
    int allocatedHeight = 0;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        int width = frame.cols;
        int height = frame.rows;
        int numPixels = width * height;
        int imageSize = numPixels * 3 * sizeof(unsigned char);

        // allocate once if needed
        if (d_image == nullptr || width != allocatedWidth || height != allocatedHeight)
        {
            if (d_image) cudaFree(d_image);
            if (d_green) cudaFree(d_green);

            CUDA_CHECK(cudaMalloc(&d_image, imageSize));
            CUDA_CHECK(cudaMalloc(&d_green, numPixels * sizeof(float)));

            allocatedWidth = width;
            allocatedHeight = height;
        }

        CUDA_CHECK(cudaMemcpy(d_image, frame.data, imageSize, cudaMemcpyHostToDevice));

        dim3 block2D(16, 16);
        dim3 grid2D((width + 15) / 16, (height + 15) / 16);

        ExtractGreenKernel << <grid2D, block2D >> > (d_image, d_green, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float greenSum = ReduceSum(d_green, numPixels);

        cout << "Total green channel sum: " << greenSum << endl;

        imshow("Webcam Feed", frame);

        if (waitKey(1) == 27) break;
    }

    if (d_image) cudaFree(d_image);
    if (d_green) cudaFree(d_green);

    return 0;
}