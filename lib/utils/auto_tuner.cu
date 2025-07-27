#include "../../include/fpn_kernels.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <memory>

struct KernelConfig {
    int tile_size;
    int channels_per_block;
    int threads_per_block;
    int shared_mem_size;
    int registers_per_thread;
    bool use_tensor_cores;
    float expected_occupancy;
    float measured_performance;
};

struct TuningResult {
    KernelConfig best_config;
    float performance_gflops;
    float memory_bandwidth_gb_s;
    float occupancy_percentage;
    std::string architecture_name;
};

class FPNAutoTuner {
private:
    std::unordered_map<std::string, TuningResult> tuning_cache_;
    cudaDeviceProp device_prop_;
    int device_id_;
    
public:
    FPNAutoTuner() {
        cudaGetDevice(&device_id_);
        cudaGetDeviceProperties(&device_prop_, device_id_);
    }
    
    // Architecture detection and optimization
    std::string get_architecture_name() {
        if (device_prop_.major == 8) {
            if (device_prop_.minor == 0) return "A100";
            if (device_prop_.minor == 6) return "A40/A30";
            if (device_prop_.minor == 9) return "A10/A16";
        } else if (device_prop_.major == 9) {
            if (device_prop_.minor == 0) return "H100/H200";
        } else if (device_prop_.major == 7) {
            if (device_prop_.minor == 5) return "T4/RTX20xx";
        }
        return "Unknown";
    }
    
    // Calculate theoretical occupancy
    float calculate_occupancy(const KernelConfig& config, int dynamic_shared_mem = 0) {
        const int max_threads_per_sm = device_prop_.maxThreadsPerMultiProcessor;
        const int max_blocks_per_sm = device_prop_.maxBlocksPerMultiProcessor;
        const int shared_mem_per_sm = device_prop_.sharedMemPerMultiprocessor;
        const int registers_per_sm = device_prop_.regsPerMultiprocessor;
        
        // Calculate limitations
        const int blocks_per_sm_threads = max_threads_per_sm / config.threads_per_block;
        const int blocks_per_sm_shared_mem = shared_mem_per_sm / (config.shared_mem_size + dynamic_shared_mem);
        const int blocks_per_sm_registers = registers_per_sm / (config.threads_per_block * config.registers_per_thread);
        
        const int effective_blocks_per_sm = std::min({
            blocks_per_sm_threads,
            blocks_per_sm_shared_mem,
            blocks_per_sm_registers,
            max_blocks_per_sm
        });
        
        return static_cast<float>(effective_blocks_per_sm * config.threads_per_block) / max_threads_per_sm;
    }
    
    // Generate candidate configurations based on architecture
    std::vector<KernelConfig> generate_candidates() {
        std::vector<KernelConfig> candidates;
        
        // Architecture-specific parameter ranges
        std::vector<int> tile_sizes, channel_blocks, thread_configs;
        
        if (device_prop_.major >= 8) { // Ampere/Ada Lovelace
            tile_sizes = {8, 16, 32};
            channel_blocks = {32, 64, 128, 256};
            thread_configs = {128, 256, 512};
        } else if (device_prop_.major == 7) { // Turing
            tile_sizes = {8, 16, 24};
            channel_blocks = {32, 64, 128};
            thread_configs = {128, 256, 384};
        } else { // Older architectures
            tile_sizes = {8, 16};
            channel_blocks = {32, 64};
            thread_configs = {128, 256};
        }
        
        for (int tile_size : tile_sizes) {
            for (int channels : channel_blocks) {
                for (int threads : thread_configs) {
                    KernelConfig config;
                    config.tile_size = tile_size;
                    config.channels_per_block = channels;
                    config.threads_per_block = threads;
                    
                    // Calculate shared memory requirements
                    config.shared_mem_size = tile_size * tile_size * channels * sizeof(float) * 2;
                    config.registers_per_thread = 32; // Estimated
                    config.use_tensor_cores = (device_prop_.major >= 7);
                    
                    // Skip invalid configurations
                    if (config.shared_mem_size > device_prop_.sharedMemPerBlock) continue;
                    if (threads > device_prop_.maxThreadsPerBlock) continue;
                    
                    config.expected_occupancy = calculate_occupancy(config);
                    
                    // Only consider configurations with reasonable occupancy
                    if (config.expected_occupancy >= 0.25f) {
                        candidates.push_back(config);
                    }
                }
            }
        }
        
        // Sort by expected occupancy
        std::sort(candidates.begin(), candidates.end(), 
                 [](const KernelConfig& a, const KernelConfig& b) {
                     return a.expected_occupancy > b.expected_occupancy;
                 });
        
        // Keep top 20 candidates
        if (candidates.size() > 20) {
            candidates.resize(20);
        }
        
        return candidates;
    }
    
    // Benchmark a specific configuration
    float benchmark_config(const KernelConfig& config, 
                          int batch_size, int in_channels, int out_channels,
                          int height, int width, int num_iterations = 100) {
        
        // Allocate test data
        const size_t input_size = batch_size * in_channels * height * width;
        const size_t weight_size = out_channels * in_channels;
        const size_t output_size = batch_size * out_channels * height * width;
        
        float *d_input, *d_weights, *d_bias, *d_output;
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_weights, weight_size * sizeof(float));
        cudaMalloc(&d_bias, out_channels * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, d_input, input_size);
        curandGenerateUniform(gen, d_weights, weight_size);
        curandGenerateUniform(gen, d_bias, out_channels);
        
        // Grid and block configuration
        const int tiles_x = (width + config.tile_size - 1) / config.tile_size;
        const int tiles_y = (height + config.tile_size - 1) / config.tile_size;
        const int total_tiles = tiles_x * tiles_y;
        
        dim3 grid(total_tiles, (out_channels + config.channels_per_block - 1) / config.channels_per_block, batch_size);
        dim3 block(config.tile_size, config.tile_size, config.threads_per_block / (config.tile_size * config.tile_size));
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            lateral_conv_kernel<float, 16><<<grid, block, config.shared_mem_size>>>(
                d_input, d_weights, d_bias, d_output,
                batch_size, in_channels, out_channels, height, width
            );
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; ++i) {
            lateral_conv_kernel<float, 16><<<grid, block, config.shared_mem_size>>>(
                d_input, d_weights, d_bias, d_output,
                batch_size, in_channels, out_channels, height, width
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Calculate performance metrics
        const float avg_time_ms = milliseconds / num_iterations;
        const size_t total_ops = 2ULL * batch_size * out_channels * in_channels * height * width;
        const float gflops = (total_ops / 1e9) / (avg_time_ms / 1000.0f);
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        curandDestroyGenerator(gen);
        
        return gflops;
    }
    
    // Main auto-tuning function
    TuningResult auto_tune_lateral_conv(int batch_size, int in_channels, int out_channels,
                                       int height, int width) {
        
        std::string cache_key = std::to_string(batch_size) + "_" + 
                               std::to_string(in_channels) + "_" + 
                               std::to_string(out_channels) + "_" + 
                               std::to_string(height) + "_" + 
                               std::to_string(width);
        
        // Check cache
        if (tuning_cache_.find(cache_key) != tuning_cache_.end()) {
            return tuning_cache_[cache_key];
        }
        
        std::cout << "Auto-tuning lateral convolution for configuration: " 
                  << batch_size << "x" << in_channels << "x" << height << "x" << width 
                  << " -> " << out_channels << std::endl;
        
        auto candidates = generate_candidates();
        
        TuningResult best_result;
        best_result.performance_gflops = 0.0f;
        
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& config = candidates[i];
            
            std::cout << "Testing config " << (i + 1) << "/" << candidates.size() 
                      << ": tile=" << config.tile_size 
                      << ", channels=" << config.channels_per_block
                      << ", threads=" << config.threads_per_block 
                      << ", occupancy=" << std::fixed << std::setprecision(2) 
                      << config.expected_occupancy << std::endl;
            
            try {
                float performance = benchmark_config(config, batch_size, in_channels, 
                                                   out_channels, height, width);
                
                std::cout << "  Performance: " << std::fixed << std::setprecision(2) 
                          << performance << " GFLOPS" << std::endl;
                
                if (performance > best_result.performance_gflops) {
                    best_result.best_config = config;
                    best_result.performance_gflops = performance;
                    best_result.occupancy_percentage = config.expected_occupancy * 100;
                    best_result.architecture_name = get_architecture_name();
                    
                    std::cout << "  *** New best configuration! ***" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "  Configuration failed: " << e.what() << std::endl;
                continue;
            }
        }
        
        // Cache the result
        tuning_cache_[cache_key] = best_result;
        
        std::cout << "Auto-tuning completed. Best performance: " 
                  << std::fixed << std::setprecision(2) 
                  << best_result.performance_gflops << " GFLOPS" << std::endl;
        
        return best_result;
    }
    
    // Auto-tune for multiple common configurations
    std::vector<TuningResult> auto_tune_common_sizes() {
        std::vector<TuningResult> results;
        
        // Common FPN configurations
        std::vector<std::tuple<int, int, int, int, int>> configs = {
            {1, 256, 256, 200, 304},   // ResNet-50 P2
            {1, 512, 256, 100, 152},   // ResNet-50 P3
            {1, 1024, 256, 50, 76},    // ResNet-50 P4
            {1, 2048, 256, 25, 38},    // ResNet-50 P5
            {4, 256, 256, 200, 304},   // Batch=4 P2
            {4, 512, 256, 100, 152},   // Batch=4 P3
            {8, 256, 256, 100, 152},   // Batch=8 P2
            {1, 256, 256, 400, 608},   // High-res P2
        };
        
        for (const auto& config : configs) {
            auto result = auto_tune_lateral_conv(
                std::get<0>(config), std::get<1>(config), std::get<2>(config),
                std::get<3>(config), std::get<4>(config)
            );
            results.push_back(result);
        }
        
        return results;
    }
    
    // Save tuning results to file
    void save_tuning_results(const std::string& filename) {
        std::ofstream file(filename);
        
        file << "Architecture: " << get_architecture_name() << std::endl;
        file << "GPU: " << device_prop_.name << std::endl;
        file << "Compute Capability: " << device_prop_.major << "." << device_prop_.minor << std::endl;
        file << std::endl;
        
        for (const auto& [key, result] : tuning_cache_) {
            file << "Configuration: " << key << std::endl;
            file << "  Best Performance: " << result.performance_gflops << " GFLOPS" << std::endl;
            file << "  Tile Size: " << result.best_config.tile_size << std::endl;
            file << "  Channels per Block: " << result.best_config.channels_per_block << std::endl;
            file << "  Threads per Block: " << result.best_config.threads_per_block << std::endl;
            file << "  Shared Memory: " << result.best_config.shared_mem_size << " bytes" << std::endl;
            file << "  Occupancy: " << result.occupancy_percentage << "%" << std::endl;
            file << std::endl;
        }
    }
};

// C interface for integration
extern "C" {
    TuningResult* fpn_auto_tune_lateral_conv(int batch_size, int in_channels, int out_channels,
                                            int height, int width) {
        static FPNAutoTuner tuner;
        static TuningResult result = tuner.auto_tune_lateral_conv(batch_size, in_channels, out_channels, height, width);
        return &result;
    }
    
    void fpn_auto_tune_save_results(const char* filename) {
        static FPNAutoTuner tuner;
        tuner.save_tuning_results(std::string(filename));
    }
}