#include "../../../include/fpn_kernels.h"
#include <cub/cub.cuh>

template<int BLOCK_SIZE>
__global__ void gradient_clipping_kernel(
    float* __restrict__ gradients,
    float* __restrict__ global_norm,
    float max_norm,
    size_t num_elements) {
    
    __shared__ float shared_norm_sq[BLOCK_SIZE];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    float local_norm_sq = 0.0f;
    if (idx < num_elements) {
        float grad_val = gradients[idx];
        local_norm_sq = grad_val * grad_val;
    }
    
    shared_norm_sq[tid] = local_norm_sq;
    __syncthreads();
    
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float block_norm_sq = BlockReduce(temp_storage).Sum(shared_norm_sq[tid]);
    
    if (tid == 0) {
        atomicAdd(global_norm, block_norm_sq);
    }
    
    __syncthreads();
    
    if (idx < num_elements) {
        float norm = sqrtf(*global_norm);
        if (norm > max_norm) {
            float scale_factor = max_norm / norm;
            gradients[idx] *= scale_factor;
        }
    }
}

template __global__ void gradient_clipping_kernel<256>(
    float*, float*, float, size_t);

template __global__ void gradient_clipping_kernel<512>(
    float*, float*, float, size_t);