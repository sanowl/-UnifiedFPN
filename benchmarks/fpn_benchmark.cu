#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../include/fpn_types.h"
#include "../src/core/fpn_core.cu"

class FPNBenchmark {
private:
    std::unique_ptr<UnifiedFPNCore> fpn_core_;
    std::vector<std::vector<float*>> allocated_tensors_;
    std::vector<FPNDimensions> test_dimensions_;
    
public:
    FPNBenchmark() {
        fpn_core_ = std::make_unique<UnifiedFPNCore>(1024 * 1024 * 1024); // 1GB workspace
        setup_test_configurations();
    }
    
    ~FPNBenchmark() {
        cleanup_allocated_memory();
    }
    
    void setup_test_configurations() {
        // Standard object detection configurations
        std::vector<std::vector<FPNDimensions>> configs = {
            // ResNet-50 FPN configuration
            {
                {1, 256, 200, 304},   // C2: 1/4 scale
                {1, 512, 100, 152},   // C3: 1/8 scale  
                {1, 1024, 50, 76},    // C4: 1/16 scale
                {1, 2048, 25, 38},    // C5: 1/32 scale
                {1, 2048, 13, 19}     // C6: 1/64 scale
            },
            // ResNet-101 FPN configuration (larger batch)
            {
                {4, 256, 200, 304},
                {4, 512, 100, 152},
                {4, 1024, 50, 76},
                {4, 2048, 25, 38},
                {4, 2048, 13, 19}
            },
            // High-resolution configuration
            {
                {1, 256, 400, 608},
                {1, 512, 200, 304},
                {1, 1024, 100, 152},
                {1, 2048, 50, 76},
                {1, 2048, 25, 38}
            },
            // Mobile/edge configuration (smaller)
            {
                {1, 128, 100, 152},
                {1, 256, 50, 76},
                {1, 512, 25, 38},
                {1, 1024, 13, 19},
                {1, 1024, 7, 10}
            }
        };
        
        test_dimensions_ = configs;
    }
    
    void allocate_test_data(const std::vector<FPNDimensions>& dims) {
        cleanup_allocated_memory();
        
        std::vector<float*> current_tensors;
        
        for (const auto& dim : dims) {
            size_t num_elements = dim.batch_size * dim.channels * dim.height * dim.width;
            float* tensor;
            
            CUDA_CHECK(cudaMalloc(&tensor, num_elements * sizeof(float)));
            
            // Initialize with random data
            std::vector<float> host_data(num_elements);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            for (auto& val : host_data) {
                val = dist(gen);
            }
            
            CUDA_CHECK(cudaMemcpy(tensor, host_data.data(), 
                                 num_elements * sizeof(float), cudaMemcpyHostToDevice));
            
            current_tensors.push_back(tensor);
        }
        
        allocated_tensors_.push_back(current_tensors);
    }
    
    void cleanup_allocated_memory() {
        for (auto& tensor_set : allocated_tensors_) {
            for (float* tensor : tensor_set) {
                cudaFree(tensor);
            }
        }
        allocated_tensors_.clear();
    }
    
    struct BenchmarkResult {
        std::string config_name;
        float avg_forward_time_ms;
        float memory_bandwidth_gb_s;
        float compute_utilization;
        size_t peak_memory_mb;
        float throughput_fps;
        std::vector<float> kernel_times_ms;
    };
    
    BenchmarkResult benchmark_configuration(const std::string& config_name,
                                           const std::vector<FPNDimensions>& backbone_dims,
                                           int num_iterations = 100) {
        
        std::cout << "Benchmarking " << config_name << "..." << std::endl;
        
        // Setup FPN for this configuration
        const int output_channels = 256;
        fpn_core_->configure_levels(backbone_dims, output_channels);
        
        std::vector<int> backbone_channels;
        backbone_channels.push_back(0); // C1 not used
        for (const auto& dim : backbone_dims) {
            backbone_channels.push_back(dim.channels);
        }
        
        fpn_core_->allocate_weights(backbone_channels, output_channels);
        
        // Allocate test data
        allocate_test_data(backbone_dims);
        
        // Create FPN tensors
        std::vector<FPNTensor<float>> backbone_features(5);
        std::vector<FPNTensor<float>> output_features(4);
        
        for (int i = 0; i < 5; ++i) {
            backbone_features[i].data = allocated_tensors_.back()[i];
            backbone_features[i].dims = backbone_dims[i];
            backbone_features[i].dtype = FPNDataType::FLOAT32;
            backbone_features[i].is_device_memory = true;
        }
        
        // Allocate output tensors
        std::vector<float*> output_ptrs;
        for (int level = 0; level < 4; ++level) {
            auto out_dim = backbone_dims[level];
            out_dim.channels = output_channels;
            
            size_t num_elements = out_dim.batch_size * out_dim.channels * out_dim.height * out_dim.width;
            float* output_ptr;
            CUDA_CHECK(cudaMalloc(&output_ptr, num_elements * sizeof(float)));
            output_ptrs.push_back(output_ptr);
            
            output_features[level].data = output_ptr;
            output_features[level].dims = out_dim;
            output_features[level].dtype = FPNDataType::FLOAT32;
            output_features[level].is_device_memory = true;
        }
        
        // Warmup runs
        for (int i = 0; i < 10; ++i) {
            fpn_core_->forward(backbone_features, output_features);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark runs
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            fpn_core_->forward(backbone_features, output_features);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float total_time_ms = duration.count() / 1000.0f;
        float avg_time_ms = total_time_ms / num_iterations;
        
        // Get performance metrics
        const auto& metrics = fpn_core_->get_performance_metrics();
        
        // Calculate throughput
        float throughput_fps = (backbone_dims[0].batch_size * 1000.0f) / avg_time_ms;
        
        // Calculate memory usage
        size_t peak_memory_mb = fpn_core_->get_memory_usage() / (1024 * 1024);
        
        // Cleanup output tensors
        for (float* ptr : output_ptrs) {
            cudaFree(ptr);
        }
        
        BenchmarkResult result;
        result.config_name = config_name;
        result.avg_forward_time_ms = avg_time_ms;
        result.memory_bandwidth_gb_s = metrics.memory_bandwidth_gb_s;
        result.compute_utilization = metrics.compute_utilization;
        result.peak_memory_mb = peak_memory_mb;
        result.throughput_fps = throughput_fps;
        result.kernel_times_ms = std::vector<float>(metrics.kernel_times_ms, 
                                                   metrics.kernel_times_ms + 8);
        
        return result;
    }
    
    void run_comprehensive_benchmark() {
        std::cout << "=== Unified FPN CUDA Kernel Benchmark ===" << std::endl;
        std::cout << std::endl;
        
        // Print GPU information
        print_gpu_info();
        
        std::vector<BenchmarkResult> results;
        
        std::vector<std::string> config_names = {
            "ResNet-50 FPN (1x)",
            "ResNet-101 FPN (4x batch)",
            "High-Resolution FPN",
            "Mobile/Edge FPN"
        };
        
        for (size_t i = 0; i < test_dimensions_.size(); ++i) {
            auto result = benchmark_configuration(config_names[i], test_dimensions_[i]);
            results.push_back(result);
        }
        
        print_benchmark_results(results);
        
        // Detailed kernel analysis
        analyze_kernel_performance(results);
        
        // Memory bandwidth analysis
        analyze_memory_performance(results);
        
        // Comparison with theoretical peak
        compare_with_theoretical_peak(results);
    }
    
private:
    void print_gpu_info() {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        std::cout << "GPU Information:" << std::endl;
        std::cout << "  Device: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        
        float memory_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        std::cout << "  Theoretical Memory Bandwidth: " << memory_bandwidth << " GB/s" << std::endl;
        std::cout << std::endl;
    }
    
    void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
        std::cout << "=== Benchmark Results ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Configuration"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Throughput"
                  << std::setw(15) << "Memory BW"
                  << std::setw(15) << "Compute %"
                  << std::setw(15) << "Memory (MB)" << std::endl;
        std::cout << std::string(100, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(25) << result.config_name
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.avg_forward_time_ms
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_fps << " FPS"
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.memory_bandwidth_gb_s << " GB/s"
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.compute_utilization * 100 << "%"
                      << std::setw(15) << result.peak_memory_mb << " MB" << std::endl;
        }
        std::cout << std::endl;
    }
    
    void analyze_kernel_performance(const std::vector<BenchmarkResult>& results) {
        std::cout << "=== Kernel Performance Analysis ===" << std::endl;
        
        std::vector<std::string> kernel_names = {
            "Lateral Conv P2", "Lateral Conv P3", "Lateral Conv P4", "Lateral Conv P5",
            "Upsample P2", "Upsample P3", "Upsample P4", "Output Conv"
        };
        
        for (const auto& result : results) {
            std::cout << result.config_name << " - Kernel Breakdown:" << std::endl;
            
            float total_kernel_time = 0;
            for (float time : result.kernel_times_ms) {
                total_kernel_time += time;
            }
            
            for (size_t i = 0; i < result.kernel_times_ms.size() && i < kernel_names.size(); ++i) {
                float percentage = (result.kernel_times_ms[i] / total_kernel_time) * 100;
                std::cout << "  " << std::left << std::setw(20) << kernel_names[i]
                          << std::setw(10) << std::fixed << std::setprecision(3) << result.kernel_times_ms[i] << " ms"
                          << std::setw(10) << std::fixed << std::setprecision(1) << percentage << "%" << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    void analyze_memory_performance(const std::vector<BenchmarkResult>& results) {
        std::cout << "=== Memory Performance Analysis ===" << std::endl;
        
        // Get theoretical peak bandwidth
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        float theoretical_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        
        for (const auto& result : results) {
            float efficiency = (result.memory_bandwidth_gb_s / theoretical_bandwidth) * 100;
            
            std::cout << result.config_name << ":" << std::endl;
            std::cout << "  Achieved Bandwidth: " << result.memory_bandwidth_gb_s << " GB/s" << std::endl;
            std::cout << "  Theoretical Peak: " << theoretical_bandwidth << " GB/s" << std::endl;
            std::cout << "  Memory Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
            std::cout << "  Peak Memory Usage: " << result.peak_memory_mb << " MB" << std::endl;
            std::cout << std::endl;
        }
    }
    
    void compare_with_theoretical_peak(const std::vector<BenchmarkResult>& results) {
        std::cout << "=== Performance vs Theoretical Peak ===" << std::endl;
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        // Estimate theoretical peak performance
        float peak_ops_per_sec = prop.clockRate * 1000.0f * prop.multiProcessorCount * 64; // Rough estimate
        
        for (const auto& result : results) {
            // Estimate operations per forward pass (rough calculation)
            size_t total_ops = 0;
            
            // This is a simplified calculation - in reality would need detailed FLOP counting
            float estimated_gflops = total_ops / (result.avg_forward_time_ms * 1e6);
            float compute_efficiency = (estimated_gflops / (peak_ops_per_sec / 1e9)) * 100;
            
            std::cout << result.config_name << ":" << std::endl;
            std::cout << "  Estimated Performance: " << estimated_gflops << " GFLOPS" << std::endl;
            std::cout << "  Compute Efficiency: " << std::fixed << std::setprecision(1) << compute_efficiency << "%" << std::endl;
            std::cout << "  Bottleneck: " << (result.memory_bandwidth_gb_s < 200 ? "Memory Bound" : "Compute Bound") << std::endl;
            std::cout << std::endl;
        }
    }
};

// Command line interface
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --iterations N    Number of benchmark iterations (default: 100)" << std::endl;
    std::cout << "  --profile         Enable CUDA profiling" << std::endl;
    std::cout << "  --config NAME     Run specific configuration only" << std::endl;
    std::cout << "  --help            Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    int num_iterations = 100;
    bool enable_profiling = false;
    std::string specific_config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--iterations" && i + 1 < argc) {
            num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--profile") {
            enable_profiling = true;
        } else if (arg == "--config" && i + 1 < argc) {
            specific_config = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Set device
    CUDA_CHECK(cudaSetDevice(0));
    
    try {
        FPNBenchmark benchmark;
        
        if (enable_profiling) {
            std::cout << "CUDA profiling enabled. Use nvprof or Nsight to collect detailed metrics." << std::endl;
            cudaProfilerStart();
        }
        
        benchmark.run_comprehensive_benchmark();
        
        if (enable_profiling) {
            cudaProfilerStop();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Benchmark completed successfully!" << std::endl;
    return 0;
}