#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                       \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            std::cerr << "CUDA Error: "                         \
                      << cudaGetErrorString(error) << std::endl; \
            return 1;                                          \
        }                                                      \
    } while (0)

// CUDA kernel for RGB grayscale conversion + 3x3 blur
__global__ void processImage(const uchar3* input,
                             uchar3* output,
                             int width,
                             int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Keep boundary pixels unchanged
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = input[y * width + x];
        return;
    }

    float sumR = 0.0f;
    float sumG = 0.0f;
    float sumB = 0.0f;

    // 3x3 box blur
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uchar3 pixel = input[(y + dy) * width + (x + dx)];

            sumR += pixel.x;
            sumG += pixel.y;
            sumB += pixel.z;
        }
    }

    uchar3 result;
    result.x = static_cast<unsigned char>(sumR / 9.0f);
    result.y = static_cast<unsigned char>(sumG / 9.0f);
    result.z = static_cast<unsigned char>(sumB / 9.0f);

    output[y * width + x] = result;
}

// Process one image using the GPU
bool processImageGPU(const std::string& inputPath,
                     const std::string& outputPath) {
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Could not read: " << inputPath << std::endl;
        return false;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(uchar3);

    uchar3* d_input = nullptr;
    uchar3* d_output = nullptr;

    cudaError_t error;

    error = cudaMalloc(&d_input, imageSize);
    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    error = cudaMalloc(&d_output, imageSize);
    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(d_input);
        return false;
    }

    error = cudaMemcpy(
        d_input,
        image.data,
        imageSize,
        cudaMemcpyHostToDevice);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y);

    processImage<<<blocks, threads>>>(
        d_input,
        d_output,
        width,
        height);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(error) << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel execution failed: "
                  << cudaGetErrorString(error) << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    cv::Mat output(height, width, CV_8UC3);

    error = cudaMemcpy(
        output.data,
        d_output,
        imageSize,
        cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    if (error != cudaSuccess) {
        std::cerr << "Copy back failed: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    if (!cv::imwrite(outputPath, output)) {
        std::cerr << "Could not write: " << outputPath << std::endl;
        return false;
    }

    return true;
}

// Generate sample images
void generateSampleImages(const std::string& directory, int count) {
    fs::create_directories(directory);

    for (int i = 0; i < count; i++) {
        int width = 256;
        int height = 256;

        cv::Mat image(height, width, CV_8UC3);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.at<cv::Vec3b>(y, x)[0] =
                    static_cast<unsigned char>((x + i * 5) % 256);

                image.at<cv::Vec3b>(y, x)[1] =
                    static_cast<unsigned char>((y + i * 3) % 256);

                image.at<cv::Vec3b>(y, x)[2] =
                    static_cast<unsigned char>((x + y + i * 7) % 256);
            }
        }

        std::string filename =
            directory + "/image_" + std::to_string(i) + ".png";

        cv::imwrite(filename, image);
    }
}

int main(int argc, char* argv[]) {
    std::string inputDirectory = "input_images";
    std::string outputDirectory = "output_images";
    int imageCount = 100;

    if (argc > 1) {
        imageCount = std::stoi(argv[1]);
    }

    fs::create_directories(inputDirectory);
    fs::create_directories(outputDirectory);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA GPU detected." << std::endl;
        return 1;
    }

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    std::cout << "GPU: " << properties.name << std::endl;
    std::cout << "CUDA devices: " << deviceCount << std::endl;
    std::cout << "Generating " << imageCount
              << " sample images..." << std::endl;

    generateSampleImages(inputDirectory, imageCount);

    int processed = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& entry : fs::directory_iterator(inputDirectory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension = entry.path().extension().string();

        if (extension != ".png" &&
            extension != ".jpg" &&
            extension != ".jpeg") {
            continue;
        }

        std::string inputPath = entry.path().string();

        std::string outputPath =
            outputDirectory + "/" +
            entry.path().stem().string() +
            "_processed.png";

        if (processImageGPU(inputPath, outputPath)) {
            processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Images processed: "
              << processed << std::endl;

    std::cout << "Processing time: "
              << elapsed << " seconds" << std::endl;

    std::cout << "GPU image processing completed successfully."
              << std::endl;

    return 0;
}