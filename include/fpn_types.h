#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>

// NCCL error checking
#define NCCL_CHECK(call) \
    do { \
        ncclResult_t res = call; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA vector types for advanced operations
struct __align__(32) half8 {
    half x[8];
};

struct __align__(64) float16 {
    float x[16];
};

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define SHARED_MEM_SIZE 49152
#define MEMORY_ALIGNMENT 256
#define L2_CACHE_LINE_SIZE 128

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Advanced CUDA intrinsics for optimal performance
#define CUDA_SAFE_CALL(call) CUDA_CHECK(call)
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

enum class FPNDataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT8,
    INT4,
    TFLOAT32  // Tensor Float 32 for Ampere
};

enum class FPNPrecisionMode {
    FULL_PRECISION,      // FP32 throughout
    MIXED_PRECISION,     // FP16 compute, FP32 accumulate
    AUTOMATIC_MIXED,     // Automatic loss scaling
    INT8_QUANTIZED,      // Post-training quantization
    DYNAMIC_QUANTIZED    // Dynamic quantization
};

enum class FPNActivation {
    NONE,
    RELU,
    LEAKY_RELU,
    SWISH
};

struct FPNDimensions {
    int batch_size;
    int channels;
    int height;
    int width;
    size_t stride_bytes;
};

struct FPNLevelConfig {
    FPNDimensions input_dim;
    FPNDimensions output_dim;
    float scale_factor;
    bool enable_output_conv;
};

template<typename T>
struct FPNTensor {
    T* data;
    FPNDimensions dims;
    FPNDataType dtype;
    bool is_device_memory;
    
    size_t total_elements() const {
        return dims.batch_size * dims.channels * dims.height * dims.width;
    }
    
    size_t size_bytes() const {
        size_t element_size = (dtype == FPNDataType::FLOAT32) ? 4 : 
                             (dtype == FPNDataType::FLOAT16) ? 2 : 1;
        return total_elements() * element_size;
    }
};

struct FPNKernelConfig {
    int tile_size;
    int channels_per_thread;
    int shared_mem_size;
    bool use_tensor_cores;
    bool enable_async_copy;
    FPNActivation activation;
    
    // Advanced configuration
    FPNPrecisionMode precision_mode;
    int max_dynamic_shared_memory;
    int preferred_occupancy_percentage;
    bool enable_persistent_threads;
    bool use_cooperative_groups;
    bool enable_ldg_caching;
    
    // Multi-GPU settings
    bool enable_multi_gpu;
    int num_gpus;
    bool use_nvlink;
    
    // Performance tuning
    int unroll_factor;
    bool vectorize_loads;
    bool use_texture_memory;
    bool enable_graph_capture;
};

struct FPNWeights {
    float* lateral_conv_weights[4];     // 1x1 convolutions
    float* lateral_conv_bias[4];
    float* output_conv_weights[4];      // 3x3 convolutions  
    float* output_conv_bias[4];
    float* batch_norm_weight[4];
    float* batch_norm_bias[4];
    float* batch_norm_mean[4];
    float* batch_norm_var[4];
    
    size_t lateral_weight_sizes[4];
    size_t output_weight_sizes[4];
    bool weights_initialized;
};

struct FPNPerformanceMetrics {
    float forward_time_ms;
    float memory_bandwidth_gb_s;
    float compute_utilization;
    size_t peak_memory_usage_bytes;
    int num_kernel_launches;
    float kernel_times_ms[8];
    
    // Advanced metrics
    float tensor_core_utilization;
    float l2_cache_hit_ratio;
    float memory_efficiency;
    float arithmetic_intensity;
    float energy_efficiency_gops_per_watt;
    std::vector<float> per_sm_utilization;
    float communication_overhead_ms;
    
    // Profiling data
    struct KernelProfile {
        std::string name;
        float duration_ms;
        size_t shared_memory_bytes;
        int registers_per_thread;
        float occupancy_percentage;
        float achieved_bandwidth_gb_s;
    };
    std::vector<KernelProfile> kernel_profiles;
};