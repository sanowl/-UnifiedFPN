#include "../../../include/fpn_kernels.h"
#include <cuda_fp16.h>

template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(256, 4)
backward_upsample_add_kernel(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_lateral,
    T* __restrict__ grad_upsampled,
    int batch_size,
    int channels,
    int out_height,
    int out_width,
    int in_height,
    int in_width,
    float scale_y,
    float scale_x) {
    
    const int batch_idx = blockIdx.z;
    const int c_base = blockIdx.y * 32;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_c = threadIdx.z;
    
    const int out_y = tile_y * TILE_SIZE + thread_y;
    const int out_x = tile_x * TILE_SIZE + thread_x;
    const int c = c_base + thread_c;
    
    if (out_y >= out_height || out_x >= out_width || c >= channels) return;
    
    const int grad_output_idx = batch_idx * channels * out_height * out_width + 
                               out_y * out_width * channels + out_x * channels + c;
    T grad_val = grad_output[grad_output_idx];
    
    const int grad_lateral_idx = batch_idx * channels * out_height * out_width + 
                                out_y * out_width * channels + out_x * channels + c;
    grad_lateral[grad_lateral_idx] = grad_val;
    
    const float src_y = (out_y + 0.5f) * scale_y - 0.5f;
    const float src_x = (out_x + 0.5f) * scale_x - 0.5f;
    
    const int y0 = __float2int_rd(src_y);
    const int x0 = __float2int_rd(src_x);
    const int y1 = min(y0 + 1, in_height - 1);
    const int x1 = min(x0 + 1, in_width - 1);
    
    const float dy = src_y - y0;
    const float dx = src_x - x0;
    
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w01 = dx * (1.0f - dy);
    const float w10 = (1.0f - dx) * dy;
    const float w11 = dx * dy;
    
    if (y0 >= 0 && y0 < in_height && x0 >= 0 && x0 < in_width) {
        int idx = batch_idx * channels * in_height * in_width + y0 * in_width * channels + x0 * channels + c;
        atomicAdd(&grad_upsampled[idx], static_cast<T>(grad_val * w00));
    }
    
    if (y0 >= 0 && y0 < in_height && x1 >= 0 && x1 < in_width) {
        int idx = batch_idx * channels * in_height * in_width + y0 * in_width * channels + x1 * channels + c;
        atomicAdd(&grad_upsampled[idx], static_cast<T>(grad_val * w01));
    }
    
    if (y1 >= 0 && y1 < in_height && x0 >= 0 && x0 < in_width) {
        int idx = batch_idx * channels * in_height * in_width + y1 * in_width * channels + x0 * channels + c;
        atomicAdd(&grad_upsampled[idx], static_cast<T>(grad_val * w10));
    }
    
    if (y1 >= 0 && y1 < in_height && x1 >= 0 && x1 < in_width) {
        int idx = batch_idx * channels * in_height * in_width + y1 * in_width * channels + x1 * channels + c;
        atomicAdd(&grad_upsampled[idx], static_cast<T>(grad_val * w11));
    }
}

template __global__ void backward_upsample_add_kernel<float, 16>(
    const float*, float*, float*, int, int, int, int, int, int, float, float);

template __global__ void backward_upsample_add_kernel<half, 16>(
    const half*, half*, half*, int, int, int, int, int, int, float, float);