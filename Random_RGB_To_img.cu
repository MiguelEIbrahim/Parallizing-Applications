#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <curand_kernel.h>

// CUDA kernel to generate random RGB values
__global__ void generateRandomImage(unsigned char* img, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        curandState state;
        curand_init(seed, idx, 0, &state);

        img[idx] = curand(&state) % 256; // Blue
        img[idx + 1] = curand(&state) % 256; // Green
        img[idx + 2] = curand(&state) % 256; // Red
    }
}

int main() {
    // Image dimensions (4K resolution: 3840x2160)
    const int width = 3840;
    const int height = 2160;

    // Create an empty image
    cv::Mat image(height, width, CV_8UC3);
    unsigned char* d_img;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_img, width * height * 3);

    // Launch the kernel to generate random RGB values
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    generateRandomImage<<<gridDim, blockDim>>>(d_img, width, height, time(0));

    // Copy the generated image from GPU to CPU
    cudaMemcpy(image.data, d_img, width * height * 3, cudaMemcpyDeviceToHost);

    // Display the image
    cv::imshow("Random Image (CUDA)", image);
    cv::waitKey(0);

    // Free GPU memory
    cudaFree(d_img);

    return 0;
}
