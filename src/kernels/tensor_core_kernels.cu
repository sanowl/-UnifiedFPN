#include "../../include/fpn_kernels.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__(256, 2)
tensor_core_lateral_conv_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weights,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    
    __shared__ __align__(16) half shared_input[WMMA_M * WMMA_K + 16];
    __shared__ __align__(16) half shared_weights[WMMA_K * WMMA_N + 16];
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int batch_idx = blockIdx.z;
    const int out_tile_y = blockIdx.y;
    const int out_tile_x = blockIdx.x;
    
    const int global_y = out_tile_y * WMMA_M;
    const int global_x = out_tile_x * WMMA_N;
    
    if (global_y >= height || global_x >= width) return;
    
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    for (int k_base = 0; k_base < in_channels; k_base += WMMA_K) {
        const int k_chunk = min(WMMA_K, in_channels - k_base);
        
        if (warp_id == 0) {
            for (int i = lane_id; i < WMMA_M * k_chunk; i += 32) {
                int y_offset = i / k_chunk;
                int k_offset = i % k_chunk;
                
                if (global_y + y_offset < height) {
                    int input_idx = batch_idx * in_channels * height * width +
                                   (global_y + y_offset) * width * in_channels +
                                   global_x * in_channels + k_base + k_offset;
                    shared_input[i] = input[input_idx];
                } else {
                    shared_input[i] = __float2half(0.0f);
                }
            }
        }
        
        if (warp_id == 1) {
            for (int i = lane_id; i < k_chunk * WMMA_N; i += 32) {
                int k_offset = i / WMMA_N;
                int n_offset = i % WMMA_N;
                
                if (global_x + n_offset < out_channels) {
                    int weight_idx = (global_x + n_offset) * in_channels + k_base + k_offset;
                    shared_weights[i] = weights[weight_idx];
                } else {
                    shared_weights[i] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag, shared_input, k_chunk);
        wmma::load_matrix_sync(b_frag, shared_weights, WMMA_N);
        
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        __syncthreads();
    }
    
    if (bias && warp_id == 0) {
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            int col = (lane_id * acc_frag.num_elements + i) % WMMA_N;
            if (global_x + col < out_channels) {
                acc_frag.x[i] = __hadd(acc_frag.x[i], bias[global_x + col]);
            }
        }
    }
    
    wmma::store_matrix_sync(&output[batch_idx * out_channels * height * width +
                                  global_y * width * out_channels + global_x * out_channels],
                           acc_frag, out_channels, wmma::mem_row_major);
}

__global__ void __launch_bounds__(1024, 1)
memory_bandwidth_benchmark_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t num_elements,
    int iterations) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = idx * 4; i + 3 < num_elements; i += stride * 4) {
            float4 data = __ldg(reinterpret_cast<const float4*>(&input[i]));
            *reinterpret_cast<float4*>(&output[i]) = data;
        }
        
        __threadfence();
    }
}

template __global__ void tensor_core_lateral_conv_kernel<16, 16, 16>(
    const half*, const half*, const half*, half*, int, int, int, int, int);

template __global__ void tensor_core_lateral_conv_kernel<32, 8, 16>(
    const half*, const half*, const half*, half*, int, int, int, int, int);

template __global__ void tensor_core_lateral_conv_kernel<8, 32, 16>(
    const half*, const half*, const half*, half*, int, int, int, int, int);