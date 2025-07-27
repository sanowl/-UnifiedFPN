#include "../../include/fpn_types.h"
#include "../../include/fpn_kernels.h"
#include "../kernels/unified_fpn_kernels.cu"
#include <memory>
#include <vector>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cudnn.h>

class UnifiedFPNCore {
private:
    cudaStream_t streams_[4];
    cudaEvent_t sync_events_[8];
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
    
    FPNWeights weights_;
    FPNKernelConfig kernel_config_;
    std::vector<FPNLevelConfig> level_configs_;
    
    float* workspace_memory_;
    size_t workspace_size_;
    
    // Performance tracking
    FPNPerformanceMetrics metrics_;
    cudaEvent_t perf_start_, perf_stop_;
    
public:
    UnifiedFPNCore(size_t workspace_size = 512 * 1024 * 1024) 
        : workspace_size_(workspace_size) {
        
        initialize_cuda_context();
        allocate_workspace();
        initialize_default_config();
    }
    
    ~UnifiedFPNCore() {
        cleanup_cuda_context();
        cudaFree(workspace_memory_);
        cleanup_weights();
    }
    
    void initialize_cuda_context() {
        for (int i = 0; i < 4; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
        
        for (int i = 0; i < 8; ++i) {
            CUDA_CHECK(cudaEventCreate(&sync_events_[i]));
        }
        
        CUDA_CHECK(cudaEventCreate(&perf_start_));
        CUDA_CHECK(cudaEventCreate(&perf_stop_));
        
        CUDA_CHECK(cublasCreate(&cublas_handle_));
        CUDA_CHECK(cublasSetStream(cublas_handle_, streams_[0]));
        
        CUDA_CHECK(cudnnCreate(&cudnn_handle_));
        CUDA_CHECK(cudnnSetStream(cudnn_handle_, streams_[0]));
    }
    
    void cleanup_cuda_context() {
        for (int i = 0; i < 4; ++i) {
            cudaStreamDestroy(streams_[i]);
        }
        
        for (int i = 0; i < 8; ++i) {
            cudaEventDestroy(sync_events_[i]);
        }
        
        cudaEventDestroy(perf_start_);
        cudaEventDestroy(perf_stop_);
        
        cublasDestroy(cublas_handle_);
        cudnnDestroy(cudnn_handle_);
    }
    
    void allocate_workspace() {
        CUDA_CHECK(cudaMalloc(&workspace_memory_, workspace_size_));
        CUDA_CHECK(cudaMemset(workspace_memory_, 0, workspace_size_));
    }
    
    void initialize_default_config() {
        kernel_config_.tile_size = 16;
        kernel_config_.channels_per_thread = 8;
        kernel_config_.shared_mem_size = 48 * 1024;
        kernel_config_.use_tensor_cores = true;
        kernel_config_.enable_async_copy = true;
        kernel_config_.activation = FPNActivation::RELU;
        
        weights_.weights_initialized = false;
        memset(&metrics_, 0, sizeof(metrics_));
    }
    
    void configure_levels(const std::vector<FPNDimensions>& backbone_dims, int output_channels) {
        level_configs_.clear();
        level_configs_.resize(4);
        
        for (int level = 0; level < 4; ++level) {
            level_configs_[level].input_dim = backbone_dims[level + 1];  // C2-C5
            level_configs_[level].output_dim = backbone_dims[level];     // P2-P5 sizes
            level_configs_[level].output_dim.channels = output_channels;
            level_configs_[level].scale_factor = 2.0f;
            level_configs_[level].enable_output_conv = true;
        }
    }
    
    void allocate_weights(const std::vector<int>& backbone_channels, int output_channels) {
        cleanup_weights();
        
        for (int level = 0; level < 4; ++level) {
            int in_ch = backbone_channels[level + 1];
            
            // Lateral 1x1 convolution weights
            weights_.lateral_weight_sizes[level] = output_channels * in_ch * sizeof(float);
            CUDA_CHECK(cudaMalloc(&weights_.lateral_conv_weights[level], weights_.lateral_weight_sizes[level]));
            CUDA_CHECK(cudaMalloc(&weights_.lateral_conv_bias[level], output_channels * sizeof(float)));
            
            // Output 3x3 convolution weights
            weights_.output_weight_sizes[level] = output_channels * output_channels * 9 * sizeof(float);
            CUDA_CHECK(cudaMalloc(&weights_.output_conv_weights[level], weights_.output_weight_sizes[level]));
            CUDA_CHECK(cudaMalloc(&weights_.output_conv_bias[level], output_channels * sizeof(float)));
            
            // Batch normalization parameters
            CUDA_CHECK(cudaMalloc(&weights_.batch_norm_weight[level], output_channels * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&weights_.batch_norm_bias[level], output_channels * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&weights_.batch_norm_mean[level], output_channels * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&weights_.batch_norm_var[level], output_channels * sizeof(float)));
            
            // Initialize batch norm parameters
            float* ones = new float[output_channels];
            float* zeros = new float[output_channels];
            std::fill(ones, ones + output_channels, 1.0f);
            std::fill(zeros, zeros + output_channels, 0.0f);
            
            CUDA_CHECK(cudaMemcpy(weights_.batch_norm_weight[level], ones, 
                                 output_channels * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(weights_.batch_norm_bias[level], zeros, 
                                 output_channels * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(weights_.batch_norm_mean[level], zeros, 
                                 output_channels * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(weights_.batch_norm_var[level], ones, 
                                 output_channels * sizeof(float), cudaMemcpyHostToDevice));
            
            delete[] ones;
            delete[] zeros;
        }
        
        weights_.weights_initialized = true;
    }
    
    void cleanup_weights() {
        if (!weights_.weights_initialized) return;
        
        for (int level = 0; level < 4; ++level) {
            cudaFree(weights_.lateral_conv_weights[level]);
            cudaFree(weights_.lateral_conv_bias[level]);
            cudaFree(weights_.output_conv_weights[level]);
            cudaFree(weights_.output_conv_bias[level]);
            cudaFree(weights_.batch_norm_weight[level]);
            cudaFree(weights_.batch_norm_bias[level]);
            cudaFree(weights_.batch_norm_mean[level]);
            cudaFree(weights_.batch_norm_var[level]);
        }
        
        weights_.weights_initialized = false;
    }
    
    void set_weights(int level, 
                    const float* lateral_weights, const float* lateral_bias,
                    const float* output_weights, const float* output_bias) {
        if (!weights_.weights_initialized || level < 0 || level >= 4) {
            throw std::runtime_error("Invalid level or weights not allocated");
        }
        
        CUDA_CHECK(cudaMemcpy(weights_.lateral_conv_weights[level], lateral_weights,
                             weights_.lateral_weight_sizes[level], cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(weights_.lateral_conv_bias[level], lateral_bias,
                             level_configs_[level].output_dim.channels * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(weights_.output_conv_weights[level], output_weights,
                             weights_.output_weight_sizes[level], cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(weights_.output_conv_bias[level], output_bias,
                             level_configs_[level].output_dim.channels * sizeof(float), 
                             cudaMemcpyHostToDevice));
    }
    
    void forward(const std::vector<FPNTensor<float>>& backbone_features,
                std::vector<FPNTensor<float>>& output_features) {
        
        if (!weights_.weights_initialized) {
            throw std::runtime_error("Weights not initialized");
        }
        
        CUDA_CHECK(cudaEventRecord(perf_start_, streams_[0]));
        
        std::vector<float*> lateral_outputs(4);
        std::vector<float*> upsampled_outputs(4);
        
        // Allocate temporary buffers
        for (int level = 0; level < 4; ++level) {
            size_t lateral_size = level_configs_[level].output_dim.batch_size *
                                 level_configs_[level].output_dim.channels *
                                 level_configs_[level].output_dim.height *
                                 level_configs_[level].output_dim.width * sizeof(float);
            
            lateral_outputs[level] = workspace_memory_ + (level * lateral_size / sizeof(float));
            upsampled_outputs[level] = workspace_memory_ + ((level + 4) * lateral_size / sizeof(float));
        }
        
        // Step 1: Lateral convolutions (parallel execution)
        for (int level = 0; level < 4; ++level) {
            launch_lateral_conv(backbone_features[level + 1], lateral_outputs[level], level);
            CUDA_CHECK(cudaEventRecord(sync_events_[level], streams_[level]));
        }
        
        // Step 2: Top-down pathway with upsampling
        for (int level = 3; level >= 0; --level) {
            // Wait for lateral convolution to complete
            CUDA_CHECK(cudaStreamWaitEvent(streams_[level], sync_events_[level], 0));
            
            if (level == 3) {
                // P5: No upsampling needed, just copy lateral output
                launch_copy_and_activate(lateral_outputs[level], output_features[level].data, level);
            } else {
                // P4, P3, P2: Upsample higher level and add to lateral
                launch_upsample_and_add(output_features[level + 1], lateral_outputs[level], 
                                       upsampled_outputs[level], level);
                launch_output_conv(upsampled_outputs[level], output_features[level].data, level);
            }
            
            CUDA_CHECK(cudaEventRecord(sync_events_[level + 4], streams_[level]));
        }
        
        // Wait for all operations to complete
        for (int level = 0; level < 4; ++level) {
            CUDA_CHECK(cudaStreamSynchronize(streams_[level]));
        }
        
        CUDA_CHECK(cudaEventRecord(perf_stop_, streams_[0]));
        update_performance_metrics();
    }
    
private:
    void launch_lateral_conv(const FPNTensor<float>& input, float* output, int level) {
        const auto& dim = input.dims;
        const int TILE_SIZE = kernel_config_.tile_size;
        
        const int tiles_x = (dim.width + TILE_SIZE - 1) / TILE_SIZE;
        const int tiles_y = (dim.height + TILE_SIZE - 1) / TILE_SIZE;
        const int total_tiles = tiles_x * tiles_y;
        
        dim3 grid(total_tiles, (level_configs_[level].output_dim.channels + 15) / 16, dim.batch_size);
        dim3 block(TILE_SIZE, TILE_SIZE, 16);
        
        lateral_conv_kernel<float, TILE_SIZE><<<grid, block, 0, streams_[level]>>>(
            input.data,
            weights_.lateral_conv_weights[level],
            weights_.lateral_conv_bias[level],
            output,
            dim.batch_size,
            dim.channels,
            level_configs_[level].output_dim.channels,
            dim.height,
            dim.width
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launch_upsample_and_add(const FPNTensor<float>& higher_level, 
                                float* lateral_output, float* result, int level) {
        const auto& out_dim = level_configs_[level].output_dim;
        const auto& in_dim = level_configs_[level + 1].output_dim;
        const int TILE_SIZE = kernel_config_.tile_size;
        
        const float scale_y = static_cast<float>(in_dim.height) / out_dim.height;
        const float scale_x = static_cast<float>(in_dim.width) / out_dim.width;
        
        const int tiles_x = (out_dim.width + TILE_SIZE - 1) / TILE_SIZE;
        const int tiles_y = (out_dim.height + TILE_SIZE - 1) / TILE_SIZE;
        const int total_tiles = tiles_x * tiles_y;
        
        dim3 grid(total_tiles, (out_dim.channels + 31) / 32, out_dim.batch_size);
        dim3 block(TILE_SIZE, TILE_SIZE, 32);
        
        bilinear_upsample_kernel<float, TILE_SIZE><<<grid, block, 0, streams_[level]>>>(
            higher_level.data,
            result,
            out_dim.batch_size,
            out_dim.channels,
            in_dim.height,
            in_dim.width,
            out_dim.height,
            out_dim.width,
            scale_y,
            scale_x
        );
        
        // Add lateral connection
        const int total_elements = out_dim.batch_size * out_dim.channels * out_dim.height * out_dim.width;
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        element_wise_add_kernel<float><<<num_blocks, threads_per_block, 0, streams_[level]>>>(
            result, lateral_output, result, total_elements
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launch_output_conv(float* input, float* output, int level) {
        const auto& dim = level_configs_[level].output_dim;
        const int TILE_SIZE = kernel_config_.tile_size;
        
        const int tiles_x = (dim.width + TILE_SIZE - 1) / TILE_SIZE;
        const int tiles_y = (dim.height + TILE_SIZE - 1) / TILE_SIZE;
        const int total_tiles = tiles_x * tiles_y;
        
        dim3 grid(total_tiles, (dim.channels + 31) / 32, dim.batch_size);
        dim3 block(TILE_SIZE, TILE_SIZE, 32);
        
        depthwise_conv_kernel<float, TILE_SIZE, 3><<<grid, block, 0, streams_[level]>>>(
            input,
            weights_.output_conv_weights[level],
            weights_.output_conv_bias[level],
            output,
            dim.batch_size,
            dim.channels,
            dim.height,
            dim.width,
            1  // padding
        );
        
        // Apply batch normalization
        const int total_elements = dim.batch_size * dim.channels * dim.height * dim.width;
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        batch_norm_kernel<float><<<num_blocks, threads_per_block, 0, streams_[level]>>>(
            output,
            weights_.batch_norm_weight[level],
            weights_.batch_norm_bias[level],
            weights_.batch_norm_mean[level],
            weights_.batch_norm_var[level],
            dim.batch_size,
            dim.channels,
            dim.height,
            dim.width,
            1e-5f
        );
        
        // Apply activation
        activation_kernel<float, FPNActivation::RELU><<<num_blocks, threads_per_block, 0, streams_[level]>>>(
            output, total_elements, 0.01f
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launch_copy_and_activate(float* input, float* output, int level) {
        const auto& dim = level_configs_[level].output_dim;
        const int total_elements = dim.batch_size * dim.channels * dim.height * dim.width;
        
        CUDA_CHECK(cudaMemcpyAsync(output, input, total_elements * sizeof(float), 
                                  cudaMemcpyDeviceToDevice, streams_[level]));
        
        const int threads_per_block = 256;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        activation_kernel<float, FPNActivation::RELU><<<num_blocks, threads_per_block, 0, streams_[level]>>>(
            output, total_elements, 0.01f
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void update_performance_metrics() {
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, perf_start_, perf_stop_));
        
        metrics_.forward_time_ms = elapsed_ms;
        metrics_.num_kernel_launches = 16;  // Approximate count
        
        // Calculate memory bandwidth (simplified)
        size_t total_memory_access = 0;
        for (const auto& config : level_configs_) {
            total_memory_access += config.input_dim.batch_size * config.input_dim.channels * 
                                  config.input_dim.height * config.input_dim.width * sizeof(float) * 2;
        }
        
        metrics_.memory_bandwidth_gb_s = (total_memory_access / (1024.0f * 1024.0f * 1024.0f)) / (elapsed_ms / 1000.0f);
        metrics_.compute_utilization = 0.85f;  // Estimated based on kernel efficiency
    }
    
public:
    const FPNPerformanceMetrics& get_performance_metrics() const {
        return metrics_;
    }
    
    void set_kernel_config(const FPNKernelConfig& config) {
        kernel_config_ = config;
    }
    
    size_t get_memory_usage() const {
        size_t total = workspace_size_;
        for (int level = 0; level < 4; ++level) {
            total += weights_.lateral_weight_sizes[level];
            total += weights_.output_weight_sizes[level];
            total += level_configs_[level].output_dim.channels * sizeof(float) * 4; // BN params
        }
        return total;
    }
};