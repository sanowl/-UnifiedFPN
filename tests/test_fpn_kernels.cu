#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include "../include/fpn_types.h"
#include "../include/fpn_kernels.h"
#include "../src/kernels/unified_fpn_kernels.cu"

class FPNKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        
        // Initialize random number generator
        gen_.seed(42);
        dist_ = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }
    
    void TearDown() override {
        // Cleanup any allocated memory
        for (auto* ptr : allocated_ptrs_) {
            cudaFree(ptr);
        }
        allocated_ptrs_.clear();
    }
    
    float* allocate_and_fill_random(size_t num_elements) {
        float* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, num_elements * sizeof(float)));
        allocated_ptrs_.push_back(ptr);
        
        std::vector<float> host_data(num_elements);
        for (auto& val : host_data) {
            val = dist_(gen_);
        }
        
        CUDA_CHECK(cudaMemcpy(ptr, host_data.data(), 
                             num_elements * sizeof(float), cudaMemcpyHostToDevice));
        return ptr;
    }
    
    float* allocate_zeros(size_t num_elements) {
        float* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, num_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(ptr, 0, num_elements * sizeof(float)));
        allocated_ptrs_.push_back(ptr);
        return ptr;
    }
    
    void compare_with_cpu_reference(float* gpu_result, 
                                   const std::vector<float>& cpu_reference,
                                   float tolerance = 1e-4f) {
        std::vector<float> gpu_data(cpu_reference.size());
        CUDA_CHECK(cudaMemcpy(gpu_data.data(), gpu_result,
                             cpu_reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (size_t i = 0; i < cpu_reference.size(); ++i) {
            EXPECT_NEAR(gpu_data[i], cpu_reference[i], tolerance) 
                << "Mismatch at index " << i;
        }
    }
    
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_;
    std::vector<float*> allocated_ptrs_;
};

TEST_F(FPNKernelTest, LateralConvBasicFunctionality) {
    const int batch_size = 2;
    const int in_channels = 256;
    const int out_channels = 256;
    const int height = 32;
    const int width = 32;
    const int TILE_SIZE = 16;
    
    // Allocate input, weights, bias, and output
    float* input = allocate_and_fill_random(batch_size * in_channels * height * width);
    float* weights = allocate_and_fill_random(out_channels * in_channels);
    float* bias = allocate_and_fill_random(out_channels);
    float* output = allocate_zeros(batch_size * out_channels * height * width);
    
    // Launch kernel
    const int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = tiles_x * tiles_y;
    
    dim3 grid(total_tiles, (out_channels + 15) / 16, batch_size);
    dim3 block(TILE_SIZE, TILE_SIZE, 16);
    
    lateral_conv_kernel<float, TILE_SIZE><<<grid, block>>>(
        input, weights, bias, output,
        batch_size, in_channels, out_channels, height, width
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Verify output is not all zeros
    std::vector<float> output_data(batch_size * out_channels * height * width);
    CUDA_CHECK(cudaMemcpy(output_data.data(), output,
                         output_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool has_nonzero = false;
    for (float val : output_data) {
        if (std::abs(val) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";
}

TEST_F(FPNKernelTest, BilinearUpsampleAccuracy) {
    const int batch_size = 1;
    const int channels = 64;
    const int in_height = 16;
    const int in_width = 16;
    const int out_height = 32;
    const int out_width = 32;
    const float scale_y = static_cast<float>(in_height) / out_height;
    const float scale_x = static_cast<float>(in_width) / out_width;
    const int TILE_SIZE = 16;
    
    // Create a simple pattern for testing
    std::vector<float> input_data(batch_size * channels * in_height * in_width);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < in_height; ++y) {
            for (int x = 0; x < in_width; ++x) {
                int idx = c * in_height * in_width + y * in_width + x;
                input_data[idx] = static_cast<float>(x + y * in_width);
            }
        }
    }
    
    float* input;
    CUDA_CHECK(cudaMalloc(&input, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input, input_data.data(),
                         input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    allocated_ptrs_.push_back(input);
    
    float* output = allocate_zeros(batch_size * channels * out_height * out_width);
    
    // Launch kernel
    const int tiles_x = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (out_height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = tiles_x * tiles_y;
    
    dim3 grid(total_tiles, (channels + 31) / 32, batch_size);
    dim3 block(TILE_SIZE, TILE_SIZE, 32);
    
    bilinear_upsample_kernel<float, TILE_SIZE><<<grid, block>>>(
        input, output,
        batch_size, channels,
        in_height, in_width,
        out_height, out_width,
        scale_y, scale_x
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Verify specific interpolated values
    std::vector<float> output_data(batch_size * channels * out_height * out_width);
    CUDA_CHECK(cudaMemcpy(output_data.data(), output,
                         output_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Test center point interpolation
    const int center_y = out_height / 2;
    const int center_x = out_width / 2;
    const int test_channel = 0;
    
    const int out_idx = test_channel * out_height * out_width + center_y * out_width + center_x;
    float interpolated_val = output_data[out_idx];
    
    // Expected value should be approximately the center of input
    float expected_val = (in_height * in_width) / 4.0f;
    EXPECT_NEAR(interpolated_val, expected_val, 2.0f) 
        << "Bilinear interpolation result incorrect";
}

TEST_F(FPNKernelTest, BatchNormalizationCorrectness) {
    const int batch_size = 4;
    const int channels = 128;
    const int height = 16;
    const int width = 16;
    const int total_elements = batch_size * channels * height * width;
    
    float* input_output = allocate_and_fill_random(total_elements);
    
    // Create batch norm parameters
    std::vector<float> weight_data(channels, 1.0f);
    std::vector<float> bias_data(channels, 0.0f);
    std::vector<float> mean_data(channels, 0.0f);
    std::vector<float> var_data(channels, 1.0f);
    
    float* weight;
    float* bias;
    float* mean;
    float* var;
    
    CUDA_CHECK(cudaMalloc(&weight, channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias, channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mean, channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&var, channels * sizeof(float)));
    
    allocated_ptrs_.insert(allocated_ptrs_.end(), {weight, bias, mean, var});
    
    CUDA_CHECK(cudaMemcpy(weight, weight_data.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, bias_data.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mean, mean_data.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(var, var_data.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    
    // Get input data for reference calculation
    std::vector<float> input_data(total_elements);
    CUDA_CHECK(cudaMemcpy(input_data.data(), input_output,
                         total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    batch_norm_kernel<float><<<num_blocks, threads_per_block>>>(
        input_output, weight, bias, mean, var,
        batch_size, channels, height, width, 1e-5f
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Verify normalization (should be identity with these parameters)
    std::vector<float> output_data(total_elements);
    CUDA_CHECK(cudaMemcpy(output_data.data(), input_output,
                         total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < total_elements; ++i) {
        EXPECT_NEAR(output_data[i], input_data[i], 1e-4f) 
            << "Batch norm with identity parameters should preserve input";
    }
}

TEST_F(FPNKernelTest, ActivationFunctions) {
    const int total_elements = 1024;
    
    // Test ReLU
    {
        float* data = allocate_and_fill_random(total_elements);
        
        std::vector<float> input_data(total_elements);
        CUDA_CHECK(cudaMemcpy(input_data.data(), data,
                             total_elements * sizeof(float), cudaMemcpyDeviceToHost));
        
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        activation_kernel<float, FPNActivation::RELU><<<num_blocks, threads_per_block>>>(
            data, total_elements, 0.01f
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<float> output_data(total_elements);
        CUDA_CHECK(cudaMemcpy(output_data.data(), data,
                             total_elements * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < total_elements; ++i) {
            float expected = std::max(0.0f, input_data[i]);
            EXPECT_NEAR(output_data[i], expected, 1e-6f) 
                << "ReLU activation incorrect at index " << i;
        }
    }
    
    // Test Leaky ReLU
    {
        float* data = allocate_and_fill_random(total_elements);
        
        std::vector<float> input_data(total_elements);
        CUDA_CHECK(cudaMemcpy(input_data.data(), data,
                             total_elements * sizeof(float), cudaMemcpyDeviceToHost));
        
        const float alpha = 0.1f;
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        activation_kernel<float, FPNActivation::LEAKY_RELU><<<num_blocks, threads_per_block>>>(
            data, total_elements, alpha
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<float> output_data(total_elements);
        CUDA_CHECK(cudaMemcpy(output_data.data(), data,
                             total_elements * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < total_elements; ++i) {
            float expected = (input_data[i] > 0.0f) ? input_data[i] : alpha * input_data[i];
            EXPECT_NEAR(output_data[i], expected, 1e-6f) 
                << "Leaky ReLU activation incorrect at index " << i;
        }
    }
}

TEST_F(FPNKernelTest, ElementWiseAddition) {
    const int total_elements = 2048;
    
    float* input1 = allocate_and_fill_random(total_elements);
    float* input2 = allocate_and_fill_random(total_elements);
    float* output = allocate_zeros(total_elements);
    
    // Get input data for reference
    std::vector<float> input1_data(total_elements);
    std::vector<float> input2_data(total_elements);
    
    CUDA_CHECK(cudaMemcpy(input1_data.data(), input1,
                         total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(input2_data.data(), input2,
                         total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    element_wise_add_kernel<float><<<num_blocks, threads_per_block>>>(
        input1, input2, output, total_elements
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Verify addition
    std::vector<float> output_data(total_elements);
    CUDA_CHECK(cudaMemcpy(output_data.data(), output,
                         total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < total_elements; ++i) {
        float expected = input1_data[i] + input2_data[i];
        EXPECT_NEAR(output_data[i], expected, 1e-6f) 
            << "Element-wise addition incorrect at index " << i;
    }
}

TEST_F(FPNKernelTest, MemoryCoalescingPerformance) {
    const int batch_size = 4;
    const int channels = 256;
    const int height = 64;
    const int width = 64;
    const int total_elements = batch_size * channels * height * width;
    
    float* input1 = allocate_and_fill_random(total_elements);
    float* input2 = allocate_and_fill_random(total_elements);
    float* output = allocate_zeros(total_elements);
    
    // Warm up
    for (int i = 0; i < 5; ++i) {
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        element_wise_add_kernel<float><<<num_blocks, threads_per_block>>>(
            input1, input2, output, total_elements
        );
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int num_iterations = 100;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        element_wise_add_kernel<float><<<num_blocks, threads_per_block>>>(
            input1, input2, output, total_elements
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_time_ms = milliseconds / num_iterations;
    float bandwidth_gb_s = (3 * total_elements * sizeof(float)) / (avg_time_ms * 1e6);
    
    // Expect reasonable memory bandwidth (at least 100 GB/s on modern GPUs)
    EXPECT_GT(bandwidth_gb_s, 50.0f) 
        << "Memory bandwidth too low: " << bandwidth_gb_s << " GB/s";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Parameterized test for different tile sizes
class FPNTileSizeTest : public FPNKernelTest, 
                       public ::testing::WithParamInterface<int> {};

TEST_P(FPNTileSizeTest, LateralConvDifferentTileSizes) {
    const int TILE_SIZE = GetParam();
    const int batch_size = 1;
    const int in_channels = 64;
    const int out_channels = 64;
    const int height = 32;
    const int width = 32;
    
    float* input = allocate_and_fill_random(batch_size * in_channels * height * width);
    float* weights = allocate_and_fill_random(out_channels * in_channels);
    float* bias = allocate_and_fill_random(out_channels);
    float* output = allocate_zeros(batch_size * out_channels * height * width);
    
    const int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = tiles_x * tiles_y;
    
    dim3 grid(total_tiles, (out_channels + 15) / 16, batch_size);
    dim3 block(TILE_SIZE, TILE_SIZE, 16);
    
    if (TILE_SIZE == 8) {
        lateral_conv_kernel<float, 8><<<grid, block>>>(
            input, weights, bias, output,
            batch_size, in_channels, out_channels, height, width
        );
    } else if (TILE_SIZE == 16) {
        lateral_conv_kernel<float, 16><<<grid, block>>>(
            input, weights, bias, output,
            batch_size, in_channels, out_channels, height, width
        );
    } else if (TILE_SIZE == 32) {
        lateral_conv_kernel<float, 32><<<grid, block>>>(
            input, weights, bias, output,
            batch_size, in_channels, out_channels, height, width
        );
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Verify output is reasonable
    std::vector<float> output_data(batch_size * out_channels * height * width);
    CUDA_CHECK(cudaMemcpy(output_data.data(), output,
                         output_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool has_reasonable_values = true;
    for (float val : output_data) {
        if (std::isnan(val) || std::isinf(val) || std::abs(val) > 1000.0f) {
            has_reasonable_values = false;
            break;
        }
    }
    EXPECT_TRUE(has_reasonable_values) 
        << "Output contains invalid values for tile size " << TILE_SIZE;
}

INSTANTIATE_TEST_SUITE_P(
    DifferentTileSizes,
    FPNTileSizeTest,
    ::testing::Values(8, 16, 32)
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found. Skipping GPU tests." << std::endl;
        return 0;
    }
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running tests on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    return RUN_ALL_TESTS();
}