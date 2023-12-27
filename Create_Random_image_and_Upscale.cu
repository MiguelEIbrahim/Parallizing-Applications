#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <curand_kernel.h>

// CUDA kernel to upscale an image
__global__ void upscaleImage(const uchar* src, uchar* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        int srcX = x * srcWidth / dstWidth;
        int srcY = y * srcHeight / dstHeight;

        int srcIdx = (srcY * srcWidth + srcX) * 3;
        int dstIdx = (y * dstWidth + x) * 3;

        dst[dstIdx] = src[srcIdx]; // Blue
        dst[dstIdx + 1] = src[srcIdx + 1]; // Green
        dst[dstIdx + 2] = src[srcIdx + 2]; // Red
    }
}

int main() {
    // Input image dimensions (1920x1200)
    const int srcWidth = 1920;
    const int srcHeight = 1200;

    // Output image dimensions (2560x1600)
    const int dstWidth = 2560;
    const int dstHeight = 1600;

    // Load the input image
    cv::Mat srcImage = cv::imread("input_image.jpg");

    if (srcImage.empty()) {
        std::cerr << "Error: Could not load the input image." << std::endl;
        return 1;
    }

    // Create an empty output image
    cv::Mat dstImage(dstHeight, dstWidth, srcImage.type());

    // GPU memory pointers
    uchar* d_srcImage;
    uchar* d_dstImage;

    // Allocate GPU memory
    cudaMalloc((void**)&d_srcImage, srcWidth * srcHeight * 3);
    cudaMalloc((void**)&d_dstImage, dstWidth * dstHeight * 3);

    // Copy the input image from CPU to GPU
    cudaMemcpy(d_srcImage, srcImage.data, srcWidth * srcHeight * 3, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to upscale the image
    dim3 blockDim(16, 16);
    dim3 gridDim((dstWidth + blockDim.x - 1) / blockDim.x, (dstHeight + blockDim.y - 1) / blockDim.y);
    upscaleImage<<<gridDim, blockDim>>>(d_srcImage, d_dstImage, srcWidth, srcHeight, dstWidth, dstHeight);

    // Copy the upscaled image from GPU to CPU
    cudaMemcpy(dstImage.data, d_dstImage, dstWidth * dstHeight * 3, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_srcImage);
    cudaFree(d_dstImage);

    // Display the original and upscaled images
    cv::imshow("Original Image", srcImage);
    cv::imshow("Upscaled Image (CUDA)", dstImage);
    cv::waitKey(0);

    return 0;
}
