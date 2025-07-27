#include <torch/extension.h>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include "../include/fpn_types.h"
#include "../src/core/fpn_core.cu"

class PyTorchFPNWrapper {
private:
    std::unique_ptr<UnifiedFPNCore> fpn_core_;
    std::vector<FPNDimensions> backbone_dims_;
    int output_channels_;
    bool initialized_;
    
public:
    PyTorchFPNWrapper(int output_channels = 256, int workspace_size_mb = 512) 
        : output_channels_(output_channels), initialized_(false) {
        fpn_core_ = std::make_unique<UnifiedFPNCore>(workspace_size_mb * 1024 * 1024);
    }
    
    ~PyTorchFPNWrapper() = default;
    
    void initialize_weights(const std::vector<torch::Tensor>& lateral_weights,
                           const std::vector<torch::Tensor>& lateral_bias,
                           const std::vector<torch::Tensor>& output_weights,
                           const std::vector<torch::Tensor>& output_bias) {
        
        if (lateral_weights.size() != 4 || lateral_bias.size() != 4 ||
            output_weights.size() != 4 || output_bias.size() != 4) {
            throw std::runtime_error("Expected 4 weight tensors for each type (P2-P5)");
        }
        
        // Validate tensor properties
        for (int i = 0; i < 4; ++i) {
            if (!lateral_weights[i].is_cuda() || !lateral_bias[i].is_cuda() ||
                !output_weights[i].is_cuda() || !output_bias[i].is_cuda()) {
                throw std::runtime_error("All weight tensors must be on CUDA device");
            }
            
            if (lateral_weights[i].dtype() != torch::kFloat32 ||
                lateral_bias[i].dtype() != torch::kFloat32 ||
                output_weights[i].dtype() != torch::kFloat32 ||
                output_bias[i].dtype() != torch::kFloat32) {
                throw std::runtime_error("All weight tensors must be float32");
            }
        }
        
        // Extract backbone channel dimensions from lateral weights
        std::vector<int> backbone_channels;
        backbone_channels.push_back(0); // C1 (not used)
        
        for (int i = 0; i < 4; ++i) {
            auto weight_shape = lateral_weights[i].sizes();
            if (weight_shape.size() != 2) {
                throw std::runtime_error("Lateral weights must be 2D tensors [out_channels, in_channels]");
            }
            
            if (weight_shape[0] != output_channels_) {
                throw std::runtime_error("Lateral weight output channels must match FPN output channels");
            }
            
            backbone_channels.push_back(weight_shape[1]);
        }
        
        // Allocate weights in FPN core
        fpn_core_->allocate_weights(backbone_channels, output_channels_);
        
        // Set weights for each level
        for (int level = 0; level < 4; ++level) {
            fpn_core_->set_weights(
                level,
                lateral_weights[level].data_ptr<float>(),
                lateral_bias[level].data_ptr<float>(),
                output_weights[level].data_ptr<float>(),
                output_bias[level].data_ptr<float>()
            );
        }
        
        initialized_ = true;
    }
    
    std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& backbone_features) {
        if (backbone_features.size() != 5) {
            throw std::runtime_error("Expected 5 backbone feature tensors (C2-C6)");
        }
        
        // Validate input tensors
        for (const auto& tensor : backbone_features) {
            if (!tensor.is_cuda()) {
                throw std::runtime_error("All input tensors must be on CUDA device");
            }
            if (tensor.dtype() != torch::kFloat32) {
                throw std::runtime_error("All input tensors must be float32");
            }
            if (tensor.dim() != 4) {
                throw std::runtime_error("All input tensors must be 4D [N, C, H, W]");
            }
        }
        
        // Initialize on first forward pass
        if (!initialized_) {
            initialize_from_backbone_features(backbone_features);
        }
        
        // Convert PyTorch tensors to FPN tensors
        std::vector<FPNTensor<float>> fpn_inputs(5);
        for (int i = 0; i < 5; ++i) {
            fpn_inputs[i] = pytorch_to_fpn_tensor(backbone_features[i]);
        }
        
        // Allocate output tensors
        std::vector<torch::Tensor> output_tensors(4);
        std::vector<FPNTensor<float>> fpn_outputs(4);
        
        for (int level = 0; level < 4; ++level) {
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(backbone_features[0].device())
                .memory_format(torch::MemoryFormat::Contiguous);
            
            // Output dimensions: P2-P5 have same spatial size as corresponding backbone features
            output_tensors[level] = torch::empty({
                backbone_features[0].size(0),  // batch_size
                output_channels_,              // channels
                backbone_features[level].size(2),  // height
                backbone_features[level].size(3)   // width
            }, options);
            
            fpn_outputs[level] = pytorch_to_fpn_tensor(output_tensors[level]);
        }
        
        // Run FPN forward pass
        fpn_core_->forward(fpn_inputs, fpn_outputs);
        
        return output_tensors;
    }
    
    std::tuple<float, float, float> benchmark(const std::vector<torch::Tensor>& backbone_features, 
                                             int num_iterations = 100) {
        if (!initialized_) {
            // Run one forward pass to initialize
            auto _ = forward(backbone_features);
        }
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            auto _ = forward(backbone_features);
        }
        
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto _ = forward(backbone_features);
        }
        
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avg_time_ms = duration.count() / (1000.0f * num_iterations);
        
        const auto& metrics = fpn_core_->get_performance_metrics();
        float memory_bandwidth = metrics.memory_bandwidth_gb_s;
        float compute_util = metrics.compute_utilization;
        
        return std::make_tuple(avg_time_ms, memory_bandwidth, compute_util);
    }
    
    std::vector<torch::Tensor> get_performance_breakdown() {
        const auto& metrics = fpn_core_->get_performance_metrics();
        
        torch::Tensor timing_tensor = torch::from_blob(
            const_cast<float*>(metrics.kernel_times_ms), {8}, torch::kFloat32
        ).clone();
        
        torch::Tensor metrics_tensor = torch::tensor({
            metrics.forward_time_ms,
            metrics.memory_bandwidth_gb_s,
            metrics.compute_utilization,
            static_cast<float>(metrics.peak_memory_usage_bytes) / (1024.0f * 1024.0f),
            static_cast<float>(metrics.num_kernel_launches)
        }, torch::kFloat32);
        
        return {timing_tensor, metrics_tensor};
    }
    
    void set_kernel_config(int tile_size, int channels_per_thread, bool use_tensor_cores, 
                          const std::string& activation) {
        FPNKernelConfig config;
        config.tile_size = tile_size;
        config.channels_per_thread = channels_per_thread;
        config.shared_mem_size = 48 * 1024;
        config.use_tensor_cores = use_tensor_cores;
        config.enable_async_copy = true;
        
        if (activation == "relu") {
            config.activation = FPNActivation::RELU;
        } else if (activation == "leaky_relu") {
            config.activation = FPNActivation::LEAKY_RELU;
        } else if (activation == "swish") {
            config.activation = FPNActivation::SWISH;
        } else {
            config.activation = FPNActivation::NONE;
        }
        
        fpn_core_->set_kernel_config(config);
    }
    
    size_t get_memory_usage() const {
        return fpn_core_->get_memory_usage();
    }
    
private:
    void initialize_from_backbone_features(const std::vector<torch::Tensor>& backbone_features) {
        backbone_dims_.clear();
        backbone_dims_.resize(5);
        
        std::vector<int> backbone_channels;
        backbone_channels.push_back(0); // C1 not used
        
        for (int i = 0; i < 5; ++i) {
            const auto& tensor = backbone_features[i];
            backbone_dims_[i].batch_size = tensor.size(0);
            backbone_dims_[i].channels = tensor.size(1);
            backbone_dims_[i].height = tensor.size(2);
            backbone_dims_[i].width = tensor.size(3);
            backbone_dims_[i].stride_bytes = tensor.stride(0) * sizeof(float);
            
            backbone_channels.push_back(tensor.size(1));
        }
        
        fpn_core_->configure_levels(backbone_dims_, output_channels_);
        fpn_core_->allocate_weights(backbone_channels, output_channels_);
        
        // Initialize weights with Xavier/Glorot initialization
        initialize_default_weights(backbone_channels);
        
        initialized_ = true;
    }
    
    void initialize_default_weights(const std::vector<int>& backbone_channels) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int level = 0; level < 4; ++level) {
            int in_ch = backbone_channels[level + 1];
            
            // Xavier initialization for lateral weights
            float lateral_std = std::sqrt(2.0f / (in_ch + output_channels_));
            std::normal_distribution<float> lateral_dist(0.0f, lateral_std);
            
            std::vector<float> lateral_weights(output_channels_ * in_ch);
            std::vector<float> lateral_bias(output_channels_, 0.0f);
            
            for (auto& w : lateral_weights) {
                w = lateral_dist(gen);
            }
            
            // Xavier initialization for output weights (3x3 conv)
            float output_std = std::sqrt(2.0f / (9 * output_channels_));
            std::normal_distribution<float> output_dist(0.0f, output_std);
            
            std::vector<float> output_weights(output_channels_ * output_channels_ * 9);
            std::vector<float> output_bias(output_channels_, 0.0f);
            
            for (auto& w : output_weights) {
                w = output_dist(gen);
            }
            
            fpn_core_->set_weights(level, 
                                  lateral_weights.data(), lateral_bias.data(),
                                  output_weights.data(), output_bias.data());
        }
    }
    
    FPNTensor<float> pytorch_to_fpn_tensor(const torch::Tensor& tensor) {
        FPNTensor<float> fpn_tensor;
        fpn_tensor.data = tensor.data_ptr<float>();
        fpn_tensor.dims.batch_size = tensor.size(0);
        fpn_tensor.dims.channels = tensor.size(1);
        fpn_tensor.dims.height = tensor.size(2);
        fpn_tensor.dims.width = tensor.size(3);
        fpn_tensor.dims.stride_bytes = tensor.stride(0) * sizeof(float);
        fpn_tensor.dtype = FPNDataType::FLOAT32;
        fpn_tensor.is_device_memory = tensor.is_cuda();
        return fpn_tensor;
    }
};

// Standalone functions for direct usage
torch::Tensor fpn_lateral_conv(const torch::Tensor& input, const torch::Tensor& weight, 
                              const torch::Tensor& bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    auto output = torch::empty({
        input.size(0), weight.size(0), input.size(2), input.size(3)
    }, input.options());
    
    const int TILE_SIZE = 16;
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = tiles_x * tiles_y;
    
    dim3 grid(total_tiles, (out_channels + 15) / 16, batch_size);
    dim3 block(TILE_SIZE, TILE_SIZE, 16);
    
    lateral_conv_kernel<float, TILE_SIZE><<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width
    );
    
    return output;
}

torch::Tensor fpn_upsample_bilinear(const torch::Tensor& input, float scale_factor) {
    CHECK_INPUT(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_height = static_cast<int>(in_height * scale_factor);
    const int out_width = static_cast<int>(in_width * scale_factor);
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int TILE_SIZE = 16;
    const int tiles_x = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (out_height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = tiles_x * tiles_y;
    
    dim3 grid(total_tiles, (channels + 31) / 32, batch_size);
    dim3 block(TILE_SIZE, TILE_SIZE, 32);
    
    bilinear_upsample_kernel<float, TILE_SIZE><<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_height, in_width,
        out_height, out_width,
        1.0f / scale_factor, 1.0f / scale_factor
    );
    
    return output;
}

std::vector<torch::Tensor> fpn_forward_pytorch(
    const std::vector<torch::Tensor>& backbone_features,
    const std::vector<torch::Tensor>& lateral_weights,
    const std::vector<torch::Tensor>& output_weights) {
    
    static PyTorchFPNWrapper fpn_wrapper;
    return fpn_wrapper.forward(backbone_features);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    py::class_<PyTorchFPNWrapper>(m, "UnifiedFPN")
        .def(py::init<int, int>(), 
             py::arg("output_channels") = 256, 
             py::arg("workspace_size_mb") = 512)
        .def("forward", &PyTorchFPNWrapper::forward)
        .def("initialize_weights", &PyTorchFPNWrapper::initialize_weights)
        .def("benchmark", &PyTorchFPNWrapper::benchmark, 
             py::arg("backbone_features"), py::arg("num_iterations") = 100)
        .def("get_performance_breakdown", &PyTorchFPNWrapper::get_performance_breakdown)
        .def("set_kernel_config", &PyTorchFPNWrapper::set_kernel_config,
             py::arg("tile_size") = 16,
             py::arg("channels_per_thread") = 8,
             py::arg("use_tensor_cores") = true,
             py::arg("activation") = "relu")
        .def("get_memory_usage", &PyTorchFPNWrapper::get_memory_usage);
    
    m.def("fpn_lateral_conv", &fpn_lateral_conv, "FPN lateral convolution");
    m.def("fpn_upsample_bilinear", &fpn_upsample_bilinear, "FPN bilinear upsampling");
    m.def("fpn_forward_pytorch", &fpn_forward_pytorch, "Full FPN forward pass");
}