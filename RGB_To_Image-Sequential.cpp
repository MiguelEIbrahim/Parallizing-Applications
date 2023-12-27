#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include <chrono>

int main() {
    // Image dimensions (4K resolution: 3840x2160)
    const int width = 3840;
    const int height = 2160;

    // Create an empty image
    cv::Mat image(height, width, CV_8UC3);

    // Initialize a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uniform_int_distribution<int> dis(0, 255);
            cv::Vec3b pixel;
            pixel[0] = dis(gen); // Blue
            pixel[1] = dis(gen); // Green
            pixel[2] = dis(gen); // Red
            image.at<cv::Vec3b>(y, x) = pixel;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Sequential Execution Time: " << duration << " ms" << std::endl;

    // Display the image
    cv::imshow("Sequential Random Image", image);
    cv::waitKey(0);

    return 0;
}
