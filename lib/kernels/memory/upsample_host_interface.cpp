#include "../../include/fpn_kernels.h"
#include "../../include/fpn_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>

// Forward declarations for CUDA kernels
extern "C" {
    cudaError_t launch_production_upsample_float(
        const float* input, float* output,
        int batch_size, int channels, int in_height, int in_width,
        int out_height, int out_width, int align_corners,
        float* kernel_time_ms, float* bandwidth_gb_s);
    
    cudaError_t launch_production_upsample_half(
        const half* input, half* output,
        int batch_size, int channels, int in_height, int in_width,
        int out_height, int out_width, int align_corners,
        float* kernel_time_ms, float* bandwidth_gb_s);
}

// ============================================================================
// ADVANCED CUDA CONTEXT AND DEVICE MANAGEMENT
// ============================================================================

/**
 * Advanced CUDA device manager with comprehensive capabilities detection
 */
class AdvancedDeviceManager {
private:
    int device_id_;
    cudaDeviceProp props_;
    bool supports_tensor_cores_;
    bool supports_unified_memory_;
    bool supports_cooperative_groups_;
    size_t max_shared_memory_;
    
public:
    explicit AdvancedDeviceManager(int device_id = 0) : device_id_(device_id) {
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaGetDeviceProperties(&props_, device_id_));
        
        // Detect advanced features
        supports_tensor_cores_ = (props_.major >= 7); // Volta and later
        supports_unified_memory_ = (props_.unifiedAddressing != 0);
        supports_cooperative_groups_ = (props_.cooperativeLaunch != 0);
        
        // Get maximum shared memory per block
        CUDA_CHECK(cudaDeviceGetAttribute(
            reinterpret_cast<int*>(&max_shared_memory_),
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            device_id_));
    }
    
    void print_device_info() const {
        printf("=== CUDA Device Information ===\n");
        printf("Device: %s\n", props_.name);
        printf("Compute Capability: %d.%d\n", props_.major, props_.minor);
        printf("Multiprocessors: %d\n", props_.multiProcessorCount);
        printf("Global Memory: %.1f GB\n", props_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Shared Memory per Block: %zu KB\n", props_.sharedMemPerBlock / 1024);
        printf("Max Shared Memory per Block: %zu KB\n", max_shared_memory_ / 1024);
        printf("Tensor Cores: %s\n", supports_tensor_cores_ ? "Yes" : "No");
        printf("Unified Memory: %s\n", supports_unified_memory_ ? "Yes" : "No");
        printf("Cooperative Groups: %s\n", supports_cooperative_groups_ ? "Yes" : "No");
        printf("Memory Clock Rate: %.1f MHz\n", props_.memoryClockRate / 1000.0);
        printf("Memory Bus Width: %d bits\n", props_.memoryBusWidth);
        printf("Peak Memory Bandwidth: %.1f GB/s\n", 
               2.0 * props_.memoryClockRate * (props_.memoryBusWidth / 8) / 1.0e6);
        printf("===============================\n");
    }
    
    bool supports_tensor_cores() const { return supports_tensor_cores_; }
    bool supports_unified_memory() const { return supports_unified_memory_; }
    size_t max_shared_memory_per_block() const { return max_shared_memory_; }
    const cudaDeviceProp& properties() const { return props_; }
};

// ============================================================================
// MEMORY POOL WITH ADVANCED ALLOCATION STRATEGIES
// ============================================================================

/**
 * High-performance memory pool with intelligent allocation strategies
 */
template<typename T>
class AdvancedMemoryPool {
private:
    struct MemoryBlock {
        T* ptr;
        size_t size_bytes;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
        
        MemoryBlock(T* p, size_t s) : ptr(p), size_bytes(s), in_use(false),
                                     last_used(std::chrono::steady_clock::now()) {}
    };
    
    std::vector<MemoryBlock> blocks_;
    size_t total_allocated_;
    size_t peak_usage_;
    mutable std::mutex mutex_;
    
public:
    AdvancedMemoryPool() : total_allocated_(0), peak_usage_(0) {}
    
    ~AdvancedMemoryPool() {
        cleanup();
    }
    
    /**
     * Allocate memory with optimal alignment and caching
     */
    T* allocate(size_t num_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        const size_t required_bytes = num_elements * sizeof(T);
        const size_t aligned_bytes = ((required_bytes + 255) / 256) * 256; // 256-byte alignment
        
        // Try to reuse existing block
        for (auto& block : blocks_) {
            if (!block.in_use && block.size_bytes >= aligned_bytes) {
                block.in_use = true;
                block.last_used = std::chrono::steady_clock::now();
                return block.ptr;
            }
        }
        
        // Allocate new block
        T* new_ptr;
        CUDA_CHECK(cudaMalloc(&new_ptr, aligned_bytes));
        
        blocks_.emplace_back(new_ptr, aligned_bytes);
        blocks_.back().in_use = true;
        
        total_allocated_ += aligned_bytes;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        
        return new_ptr;
    }
    
    /**
     * Deallocate memory (returns to pool)
     */
    void deallocate(T* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                block.last_used = std::chrono::steady_clock::now();
                return;
            }
        }
    }
    
    /**
     * Cleanup unused blocks (garbage collection)
     */
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        const auto threshold = std::chrono::minutes(5); // 5 minutes unused
        
        for (auto it = blocks_.begin(); it != blocks_.end();) {
            if (!it->in_use && (now - it->last_used) > threshold) {
                CUDA_CHECK(cudaFree(it->ptr));
                total_allocated_ -= it->size_bytes;
                it = blocks_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    size_t total_allocated() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return total_allocated_; 
    }
    
    size_t peak_usage() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_usage_; 
    }
    
    void print_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        printf("=== Memory Pool Statistics ===\n");
        printf("Total Allocated: %.1f MB\n", total_allocated_ / (1024.0 * 1024.0));
        printf("Peak Usage: %.1f MB\n", peak_usage_ / (1024.0 * 1024.0));
        printf("Active Blocks: %zu\n", 
               std::count_if(blocks_.begin(), blocks_.end(), 
                           [](const MemoryBlock& b) { return b.in_use; }));
        printf("Total Blocks: %zu\n", blocks_.size());
        printf("==============================\n");
    }
};

// Global memory pools for different data types
static AdvancedMemoryPool<float> g_float_pool;
static AdvancedMemoryPool<half> g_half_pool;

// ============================================================================
// COMPREHENSIVE PERFORMANCE PROFILER
// ============================================================================

/**
 * Advanced performance profiler with detailed metrics collection
 */
class PerformanceProfiler {
private:
    struct KernelProfile {
        std::string name;
        std::vector<float> execution_times;
        std::vector<float> memory_bandwidths;
        std::vector<float> compute_utilizations;
        size_t total_invocations;
        
        KernelProfile(const std::string& n) : name(n), total_invocations(0) {}
        
        void add_measurement(float time_ms, float bandwidth_gb_s, float utilization) {
            execution_times.push_back(time_ms);
            memory_bandwidths.push_back(bandwidth_gb_s);
            compute_utilizations.push_back(utilization);
            total_invocations++;
        }
        
        float average_time() const {
            return execution_times.empty() ? 0.0f :
                   std::accumulate(execution_times.begin(), execution_times.end(), 0.0f) / execution_times.size();
        }
        
        float max_bandwidth() const {
            return memory_bandwidths.empty() ? 0.0f :
                   *std::max_element(memory_bandwidths.begin(), memory_bandwidths.end());
        }
        
        float average_utilization() const {
            return compute_utilizations.empty() ? 0.0f :
                   std::accumulate(compute_utilizations.begin(), compute_utilizations.end(), 0.0f) / 
                   compute_utilizations.size();
        }
    };
    
    std::map<std::string, KernelProfile> profiles_;
    mutable std::mutex mutex_;
    
public:
    void record_kernel_execution(const std::string& kernel_name, 
                                float time_ms, float bandwidth_gb_s, float utilization) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = profiles_.find(kernel_name);
        if (it == profiles_.end()) {
            profiles_.emplace(kernel_name, KernelProfile(kernel_name));
            it = profiles_.find(kernel_name);
        }
        
        it->second.add_measurement(time_ms, bandwidth_gb_s, utilization);
    }
    
    void print_summary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        printf("=== Performance Profiler Summary ===\n");
        printf("%-25s %10s %12s %15s %12s\n", 
               "Kernel", "Calls", "Avg Time(ms)", "Max BW(GB/s)", "Avg Util(%)");
        printf("--------------------------------------------------------------------------------\n");
        
        for (const auto& [name, profile] : profiles_) {
            printf("%-25s %10zu %12.3f %15.1f %12.1f\n",
                   name.c_str(), profile.total_invocations,
                   profile.average_time(), profile.max_bandwidth(),
                   profile.average_utilization() * 100.0f);
        }
        printf("=====================================\n");
    }
    
    void export_to_csv(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(filename);
        file << "Kernel,Invocations,Average_Time_ms,Max_Bandwidth_GB_s,Average_Utilization\n";
        
        for (const auto& [name, profile] : profiles_) {
            file << name << "," << profile.total_invocations << ","
                 << profile.average_time() << "," << profile.max_bandwidth() << ","
                 << profile.average_utilization() << "\n";
        }
    }
};

static PerformanceProfiler g_profiler;

// ============================================================================
// HIGH-LEVEL PRODUCTION INTERFACE
// ============================================================================

/**
 * Production-quality upsampling interface with automatic optimization
 */
class ProductionUpsamplingEngine {
private:
    AdvancedDeviceManager device_manager_;
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;
    bool initialized_;
    
public:
    ProductionUpsamplingEngine(int device_id = 0) 
        : device_manager_(device_id), initialized_(false) {
        initialize();
    }
    
    ~ProductionUpsamplingEngine() {
        cleanup();
    }
    
    void initialize() {
        if (initialized_) return;
        
        CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        CUDA_CHECK(cudaStreamCreate(&memory_stream_));
        
        device_manager_.print_device_info();
        initialized_ = true;
    }
    
    void cleanup() {
        if (!initialized_) return;
        
        CUDA_CHECK(cudaStreamDestroy(compute_stream_));
        CUDA_CHECK(cudaStreamDestroy(memory_stream_));
        
        g_float_pool.cleanup();
        g_half_pool.cleanup();
        
        initialized_ = false;
    }
    
    /**
     * High-level upsampling interface with automatic data type detection
     */
    template<typename T>
    FPNPerformanceMetrics upsample(
        const FPNTensor<T>& input,
        FPNTensor<T>& output,
        float scale_factor_y = 2.0f,
        float scale_factor_x = 2.0f,
        bool align_corners = false,
        bool enable_profiling = true) {
        
        if (!initialized_) {
            throw std::runtime_error("Engine not initialized");
        }
        
        // Validate input dimensions
        validate_tensors(input, output, scale_factor_y, scale_factor_x);
        
        // Calculate output dimensions if not set
        if (output.dims.height == 0 || output.dims.width == 0) {
            output.dims.height = static_cast<int>(input.dims.height * scale_factor_y);
            output.dims.width = static_cast<int>(input.dims.width * scale_factor_x);
        }
        
        // Performance measurement
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start, compute_stream_));
        
        // Launch appropriate kernel based on data type
        float kernel_time_ms = 0.0f;
        float bandwidth_gb_s = 0.0f;
        cudaError_t result = cudaSuccess;
        
        if constexpr (std::is_same_v<T, float>) {
            result = launch_production_upsample_float(
                input.data, output.data,
                input.dims.batch_size, input.dims.channels,
                input.dims.height, input.dims.width,
                output.dims.height, output.dims.width,
                align_corners ? 1 : 0,
                &kernel_time_ms, &bandwidth_gb_s);
        } else if constexpr (std::is_same_v<T, half>) {
            result = launch_production_upsample_half(
                input.data, output.data,
                input.dims.batch_size, input.dims.channels,
                input.dims.height, input.dims.width,
                output.dims.height, output.dims.width,
                align_corners ? 1 : 0,
                &kernel_time_ms, &bandwidth_gb_s);
        } else {
            throw std::runtime_error("Unsupported data type for upsampling");
        }
        
        CUDA_CHECK(cudaEventRecord(stop, compute_stream_));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " + 
                                   std::string(cudaGetErrorString(result)));
        }
        
        // Calculate comprehensive performance metrics
        float total_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start, stop));
        
        FPNPerformanceMetrics metrics = {};
        metrics.forward_time_ms = total_time_ms;
        metrics.memory_bandwidth_gb_s = bandwidth_gb_s;
        
        const size_t input_bytes = input.size_bytes();
        const size_t output_bytes = output.size_bytes();
        const size_t total_bytes = input_bytes + output_bytes;
        
        metrics.memory_efficiency = (bandwidth_gb_s / device_manager_.properties().memoryClockRate) * 100.0f;
        
        // Estimate compute utilization
        const size_t total_ops = output.total_elements() * 4; // 4 ops per bilinear interpolation
        const float theoretical_gflops = device_manager_.properties().clockRate * 1e-6 * 
                                       device_manager_.properties().multiProcessorCount * 64; // estimated cores per SM
        metrics.compute_utilization = (total_ops / (total_time_ms * 1e-3)) / (theoretical_gflops * 1e9);
        
        metrics.peak_memory_usage_bytes = total_bytes;
        metrics.num_kernel_launches = 1;
        metrics.kernel_times_ms[0] = kernel_time_ms;
        
        // Record profiling data
        if (enable_profiling) {
            std::string kernel_name = std::string("upsample_") + (std::is_same_v<T, float> ? "float" : "half");
            g_profiler.record_kernel_execution(kernel_name, kernel_time_ms, bandwidth_gb_s, 
                                             metrics.compute_utilization);
        }
        
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        return metrics;
    }
    
    /**
     * Batch processing interface for multiple tensors
     */
    template<typename T>
    std::vector<FPNPerformanceMetrics> upsample_batch(
        const std::vector<FPNTensor<T>>& inputs,
        std::vector<FPNTensor<T>>& outputs,
        float scale_factor_y = 2.0f,
        float scale_factor_x = 2.0f,
        bool align_corners = false) {
        
        if (inputs.size() != outputs.size()) {
            throw std::invalid_argument("Input and output batch sizes must match");
        }
        
        std::vector<FPNPerformanceMetrics> metrics_batch;
        metrics_batch.reserve(inputs.size());
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto metrics = upsample(inputs[i], outputs[i], scale_factor_y, scale_factor_x, align_corners);
            metrics_batch.push_back(metrics);
        }
        
        return metrics_batch;
    }
    
    void print_profiling_summary() const {
        g_profiler.print_summary();
    }
    
    void export_profiling_data(const std::string& filename) const {
        g_profiler.export_to_csv(filename);
    }
    
private:
    template<typename T>
    void validate_tensors(const FPNTensor<T>& input, const FPNTensor<T>& output,
                         float scale_y, float scale_x) {
        if (!input.data || !output.data) {
            throw std::invalid_argument("Null tensor data pointers");
        }
        
        if (input.dims.batch_size <= 0 || input.dims.channels <= 0 ||
            input.dims.height <= 0 || input.dims.width <= 0) {
            throw std::invalid_argument("Invalid input tensor dimensions");
        }
        
        if (scale_y <= 0.0f || scale_x <= 0.0f) {
            throw std::invalid_argument("Scale factors must be positive");
        }
        
        if (!input.is_device_memory || !output.is_device_memory) {
            throw std::invalid_argument("Tensors must be in device memory");
        }
    }
};

// ============================================================================
// C-STYLE API FOR PYTHON INTEGRATION
// ============================================================================

extern "C" {

/**
 * Create upsampling engine instance
 */
void* create_upsample_engine(int device_id) {
    try {
        return new ProductionUpsamplingEngine(device_id);
    } catch (const std::exception&) {
        return nullptr;
    }
}

/**
 * Destroy upsampling engine instance
 */
void destroy_upsample_engine(void* engine) {
    if (engine) {
        delete static_cast<ProductionUpsamplingEngine*>(engine);
    }
}

/**
 * High-level upsampling function for Python integration
 */
int production_upsample_float(
    void* engine,
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale_y,
    float scale_x,
    int align_corners,
    float* metrics_out) {
    
    try {
        auto* engine_ptr = static_cast<ProductionUpsamplingEngine*>(engine);
        if (!engine_ptr) return -1;
        
        // Create tensor wrappers
        FPNTensor<float> input_tensor = {
            const_cast<float*>(input),
            {batch_size, channels, in_height, in_width, 0},
            FPNDataType::FLOAT32,
            true
        };
        
        FPNTensor<float> output_tensor = {
            output,
            {batch_size, channels, out_height, out_width, 0},
            FPNDataType::FLOAT32,
            true
        };
        
        auto metrics = engine_ptr->upsample(input_tensor, output_tensor, scale_y, scale_x, 
                                          align_corners != 0);
        
        // Copy metrics to output array if provided
        if (metrics_out) {
            metrics_out[0] = metrics.forward_time_ms;
            metrics_out[1] = metrics.memory_bandwidth_gb_s;
            metrics_out[2] = metrics.compute_utilization;
            metrics_out[3] = static_cast<float>(metrics.peak_memory_usage_bytes);
        }
        
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

/**
 * Print profiling summary
 */
void print_upsample_profiling_summary(void* engine) {
    if (engine) {
        static_cast<ProductionUpsamplingEngine*>(engine)->print_profiling_summary();
    }
}

/**
 * Export profiling data to CSV
 */
int export_upsample_profiling_data(void* engine, const char* filename) {
    try {
        if (engine && filename) {
            static_cast<ProductionUpsamplingEngine*>(engine)->export_profiling_data(filename);
            return 0;
        }
        return -1;
    } catch (const std::exception&) {
        return -1;
    }
}

} // extern "C"