#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <atomic>
#include <chrono>
#include <exception>
#include <thread>
#include <mutex>
#include <condition_variable>

// Advanced error handling with stack traces and recovery
class FPNException : public std::exception {
public:
    FPNException(const std::string& message, const char* file, int line, const char* function)
        : message_(message), file_(file), line_(line), function_(function) {
        full_message_ = "FPN Error in " + std::string(function) + " at " + 
                       std::string(file) + ":" + std::to_string(line) + " - " + message;
    }
    
    const char* what() const noexcept override { return full_message_.c_str(); }
    const std::string& get_message() const { return message_; }
    const char* get_file() const { return file_; }
    int get_line() const { return line_; }
    const char* get_function() const { return function_; }
    
private:
    std::string message_;
    std::string full_message_;
    const char* file_;
    int line_;
    const char* function_;
};

#define FPN_THROW(message) throw FPNException(message, __FILE__, __LINE__, __func__)

// NCCL error checking with recovery
#define NCCL_CHECK(call) \
    do { \
        ncclResult_t res = call; \
        if (res != ncclSuccess) { \
            std::string error_msg = "NCCL error: " + std::string(ncclGetErrorString(res)); \
            FPN_THROW(error_msg); \
        } \
    } while(0)

// Advanced CUDA vector types with optimized alignment
struct __align__(32) half8 {
    half x[8];
    
    __device__ __host__ half8() { memset(x, 0, sizeof(x)); }
    __device__ __host__ half8(half val) { 
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = val; 
    }
    
    __device__ __forceinline__ half8 operator+(const half8& other) const {
        half8 result;
        #pragma unroll
        for (int i = 0; i < 8; ++i) result.x[i] = __hadd(x[i], other.x[i]);
        return result;
    }
};

struct __align__(64) float16 {
    float x[16];
    
    __device__ __host__ float16() { memset(x, 0, sizeof(x)); }
    __device__ __host__ float16(float val) { 
        #pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = val; 
    }
    
    __device__ __forceinline__ float16 operator+(const float16& other) const {
        float16 result;
        #pragma unroll
        for (int i = 0; i < 16; ++i) result.x[i] = x[i] + other.x[i];
        return result;
    }
};

// Architecture-specific vector types for optimal performance
struct __align__(16) bfloat16_4 {
    __nv_bfloat16 x[4];
};

struct __align__(32) int8_16 {
    int8_t x[16];
};

struct __align__(16) int4_32 {
    uint32_t packed[4]; // 32 4-bit values packed into 4 uint32_t
};

// Architecture-adaptive constants with runtime detection
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define SHARED_MEM_SIZE_STATIC 49152
#define MEMORY_ALIGNMENT 256
#define L2_CACHE_LINE_SIZE 128
#define TENSOR_CORE_TILE_SIZE 16
#define ASYNC_COPY_ALIGNMENT 16
#define GLOBAL_MEMORY_COALESCE_BYTES 128
#define PREFETCH_DISTANCE 4

// Dynamic device capabilities structure
struct DeviceCapabilities {
    int major;
    int minor;
    size_t max_shared_memory_per_block;
    size_t max_shared_memory_per_multiprocessor;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int multiprocessor_count;
    int warp_size;
    bool supports_tensor_cores;
    bool supports_cooperative_groups;
    bool supports_async_copy;
    bool supports_unified_memory;
    size_t l2_cache_size;
    int memory_bus_width;
    float memory_clock_rate;
    int compute_mode;
    
    static DeviceCapabilities detect(int device_id = -1);
};

// Production-grade CUDA error handling with detailed diagnostics
class CudaErrorHandler {
public:
    static void check(cudaError_t error, const char* file, int line, const char* func, const char* call_str) {
        if (error != cudaSuccess) {
            std::string detailed_msg = create_detailed_error_message(error, file, line, func, call_str);
            FPN_THROW(detailed_msg);
        }
    }
    
    static void check_last_error(const char* file, int line, const char* func) {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            check(error, file, line, func, "cudaGetLastError()");
        }
    }
    
private:
    static std::string create_detailed_error_message(cudaError_t error, const char* file, 
                                                   int line, const char* func, const char* call_str) {
        std::string base_msg = "CUDA Error: " + std::string(cudaGetErrorString(error));
        base_msg += "\nCall: " + std::string(call_str);
        
        // Add device information
        int device;
        if (cudaGetDevice(&device) == cudaSuccess) {
            cudaDeviceProp props;
            if (cudaGetDeviceProperties(&props, device) == cudaSuccess) {
                base_msg += "\nDevice: " + std::string(props.name);
                base_msg += " (Compute " + std::to_string(props.major) + "." + std::to_string(props.minor) + ")";
            }
            
            size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                base_msg += "\nMemory: " + std::to_string(free_mem / (1024*1024)) + "/" + 
                           std::to_string(total_mem / (1024*1024)) + " MB free";
            }
        }
        
        return base_msg;
    }
};

#define CUDA_CHECK(call) \
    CudaErrorHandler::check(call, __FILE__, __LINE__, __func__, #call)
    
#define CUDA_CHECK_LAST_ERROR() \
    CudaErrorHandler::check_last_error(__FILE__, __LINE__, __func__)

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
    TFLOAT32,        // Tensor Float 32 for Ampere
    FP8_E4M3,        // FP8 for H100+
    FP8_E5M2,        // Alternative FP8 format
    DOUBLE,          // For high-precision requirements
    INT16,           // For specific quantization schemes
    UINT8,           // Unsigned 8-bit
    COMPLEX64,       // Complex numbers
    BOOL,            // Boolean/binary
    AUTO             // Automatic type selection
};

// Data type utilities with compile-time optimization
template<FPNDataType DT>
struct DataTypeTraits {
    static_assert(DT != FPNDataType::AUTO, "AUTO type requires runtime resolution");
};

template<> struct DataTypeTraits<FPNDataType::FLOAT32> {
    using type = float;
    static constexpr size_t size_bytes = 4;
    static constexpr bool is_floating_point = true;
    static constexpr bool supports_tensor_cores = false;
    static constexpr int alignment = 4;
};

template<> struct DataTypeTraits<FPNDataType::FLOAT16> {
    using type = __half;
    static constexpr size_t size_bytes = 2;
    static constexpr bool is_floating_point = true;
    static constexpr bool supports_tensor_cores = true;
    static constexpr int alignment = 2;
};

template<> struct DataTypeTraits<FPNDataType::BFLOAT16> {
    using type = __nv_bfloat16;
    static constexpr size_t size_bytes = 2;
    static constexpr bool is_floating_point = true;
    static constexpr bool supports_tensor_cores = true;
    static constexpr int alignment = 2;
};

template<> struct DataTypeTraits<FPNDataType::INT8> {
    using type = int8_t;
    static constexpr size_t size_bytes = 1;
    static constexpr bool is_floating_point = false;
    static constexpr bool supports_tensor_cores = true;
    static constexpr int alignment = 1;
};

enum class FPNPrecisionMode {
    FULL_PRECISION,      // FP32 throughout
    MIXED_PRECISION,     // FP16 compute, FP32 accumulate
    AUTOMATIC_MIXED,     // Automatic loss scaling
    INT8_QUANTIZED,      // Post-training quantization
    DYNAMIC_QUANTIZED,   // Dynamic quantization
    BFLOAT16_MIXED,      // BF16 compute, FP32 accumulate
    FP8_EXPERIMENTAL,    // FP8 for future hardware
    ADAPTIVE_PRECISION,  // Runtime precision selection
    TENSOR_CORE_OPTIMIZED  // Optimized for tensor core usage
};

// Precision configuration with automatic optimization
struct PrecisionConfig {
    FPNPrecisionMode mode;
    float loss_scale_factor;
    bool enable_automatic_scaling;
    bool use_tensor_cores_when_available;
    int quantization_bits;
    float quantization_range_min;
    float quantization_range_max;
    bool enable_dynamic_range_adjustment;
    
    // Performance tuning
    bool prefer_speed_over_accuracy;
    float acceptable_error_threshold;
    bool enable_precision_monitoring;
    
    static PrecisionConfig create_optimized(const DeviceCapabilities& caps, 
                                           FPNDataType preferred_type = FPNDataType::AUTO);
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

// Memory pool integration for efficient memory management
class MemoryPool {
public:
    static MemoryPool& get_instance() {
        static MemoryPool instance;
        return instance;
    }
    
    void* allocate(size_t size, size_t alignment = MEMORY_ALIGNMENT);
    void deallocate(void* ptr);
    void reset();
    size_t get_total_allocated() const { return total_allocated_.load(); }
    size_t get_peak_usage() const { return peak_usage_.load(); }
    
private:
    MemoryPool() = default;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_usage_{0};
    std::mutex allocation_mutex_;
    std::vector<std::pair<void*, size_t>> allocated_blocks_;
};

// Automatic kernel configuration tuner
class AutoTuner {
public:
    static FPNKernelConfig tune_configuration(
        const FPNDimensions& input_dims,
        const FPNDimensions& output_dims,
        FPNDataType dtype,
        const DeviceCapabilities& device_caps
    );
    
    static void cache_tuning_result(
        const std::string& cache_key,
        const FPNKernelConfig& config
    );
    
    static bool load_cached_result(
        const std::string& cache_key,
        FPNKernelConfig& config
    );
    
private:
    static std::string generate_cache_key(
        const FPNDimensions& input_dims,
        const FPNDimensions& output_dims,
        FPNDataType dtype,
        const DeviceCapabilities& device_caps
    );
};