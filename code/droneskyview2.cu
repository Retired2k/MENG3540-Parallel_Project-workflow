#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

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
// =====================================================
__global__ void ExtractGreenKernel(unsigned char* image, float* green, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int imageIdx = pixelIdx * 3;

    green[pixelIdx] = (float)image[imageIdx + 1]; // Green channel
}

// =====================================================
// Kernel 2: Reduction
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

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// =====================================================
// Host Reduction Function
// =====================================================
float ReduceSum(float* d_input, int n)
{
    int threads = 256;
    int currentSize = n;

    float* d_in = d_input;
    float* d_out = nullptr;

    while (currentSize > 1)
    {
        int blocks = (currentSize + (threads * 2 - 1)) / (threads * 2);

        CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));

        ReductionKernel << <blocks, threads, threads * sizeof(float) >> > (d_in, d_out, currentSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_in != d_input)
            cudaFree(d_in);

        d_in = d_out;
        currentSize = blocks;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost));

    if (d_in != d_input)
        cudaFree(d_in);

    return result;
}

// =====================================================
// Histogram Function
// =====================================================
Mat drawColorHistogram(const Mat& frame)
{
    vector<Mat> bgr;
    split(frame, bgr);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    Mat bHist, gHist, rHist;

    calcHist(&bgr[0], 1, 0, Mat(), bHist, 1, &histSize, &histRange);
    calcHist(&bgr[1], 1, 0, Mat(), gHist, 1, &histSize, &histRange);
    calcHist(&bgr[2], 1, 0, Mat(), rHist, 1, &histSize, &histRange);

    int histW = 512, histH = 400;
    int binW = cvRound((double)histW / histSize);

    Mat histImage(histH, histW, CV_8UC3, Scalar(0, 0, 0));

    normalize(bHist, bHist, 0, histImage.rows, NORM_MINMAX);
    normalize(gHist, gHist, 0, histImage.rows, NORM_MINMAX);
    normalize(rHist, rHist, 0, histImage.rows, NORM_MINMAX);

    for (int i = 1; i < histSize; i++)
    {
        line(histImage,
            Point(binW * (i - 1), histH - cvRound(bHist.at<float>(i - 1))),
            Point(binW * i, histH - cvRound(bHist.at<float>(i))),
            Scalar(255, 0, 0), 2);

        line(histImage,
            Point(binW * (i - 1), histH - cvRound(gHist.at<float>(i - 1))),
            Point(binW * i, histH - cvRound(gHist.at<float>(i))),
            Scalar(0, 255, 0), 2);

        line(histImage,
            Point(binW * (i - 1), histH - cvRound(rHist.at<float>(i - 1))),
            Point(binW * i, histH - cvRound(rHist.at<float>(i))),
            Scalar(0, 0, 255), 2);
    }

    return histImage;
}

// =====================================================
// Main
// =====================================================
int main()
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Camera failed to open\n";
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

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        ExtractGreenKernel << <grid, block >> > (d_image, d_green, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());

        float greenSum = ReduceSum(d_green, numPixels);
        float avgGreen = greenSum / numPixels;
        float greenPercent = (avgGreen / 255.0f) * 100.0f;

        cout << "Sum: " << greenSum
            << " | Avg: " << avgGreen
            << " | Green %: " << greenPercent << "%" << endl;

        Mat hist = drawColorHistogram(frame);

        imshow("Webcam", frame);
        imshow("Histogram", hist);

        if (waitKey(1) == 27) break;
    }

    if (d_image) cudaFree(d_image);
    if (d_green) cudaFree(d_green);

    cap.release();
    destroyAllWindows();

    return 0;
}