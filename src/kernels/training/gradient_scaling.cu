#include "../../../include/fpn_kernels.h"
#include <cuda_fp16.h>

__global__ void gradient_scaling_kernel(
    half* __restrict__ gradients,
    float* __restrict__ master_gradients,
    float loss_scale,
    bool* __restrict__ overflow_detected,
    size_t num_elements) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        half grad_val = gradients[idx];
        
        if (__hisinf(grad_val) || __hisnan(grad_val)) {
            *overflow_detected = true;
            return;
        }
        
        float unscaled_grad = __half2float(grad_val) / loss_scale;
        
        if (isnan(unscaled_grad) || isinf(unscaled_grad)) {
            *overflow_detected = true;
            return;
        }
        
        master_gradients[idx] = unscaled_grad;
    }
}

__global__ void adaptive_loss_scaling_kernel(
    float* __restrict__ loss_scale,
    bool* __restrict__ overflow_detected,
    int* __restrict__ scale_window,
    int scale_window_size,
    float scale_factor,
    float backoff_factor) {
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*overflow_detected) {
            *loss_scale *= backoff_factor;
            *scale_window = 0;
            *overflow_detected = false;
        } else {
            (*scale_window)++;
            
            if (*scale_window >= scale_window_size) {
                *loss_scale *= scale_factor;
                *scale_window = 0;
            }
        }
        
        *loss_scale = fmaxf(1.0f, fminf(*loss_scale, 65536.0f));
    }
}