#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>


__device__ __forceinline__ int32_t float_as_int32_device(float f) {
    return __float_as_int(f);
}

__device__ __forceinline__ float int32_as_float_device(int32_t i) {
    return __int_as_float(i);
}

__global__ void stochastic_copy_kernel(
    at::BFloat16* __restrict__ target_ptr,
    const float* __restrict__ source_ptr,
    const int64_t n_elements,
    uint64_t seed,
    uint64_t sequence_offset)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        curandStatePhilox4_32_10_t rng_state;
        curand_init(seed, idx + sequence_offset, 0, &rng_state);
        unsigned int random_uint32 = curand(&rng_state);
        uint16_t random_uint16 = static_cast<uint16_t>(random_uint32);
        float source_val_fp32 = source_ptr[idx];
        int32_t val_int32_from_random = static_cast<int32_t>(random_uint16);
        int32_t val_int32_from_source = float_as_int32_device(source_val_fp32);
        int32_t temp_result_int32 = val_int32_from_random + val_int32_from_source;
        temp_result_int32 &= 0xFFFF0000;
        float rounded_fp32_val = int32_as_float_device(temp_result_int32);
        target_ptr[idx] = at::BFloat16(rounded_fp32_val);
    }
}


// Função C++ (launcher) - MODIFICADA para aceitar cudaStream_t
void foreach_copy_stochastic_cuda_launcher(
    const std::vector<at::Tensor>& targets_bf16,
    const std::vector<at::Tensor>& sources_fp32,
    uint64_t base_seed,
    cudaStream_t stream) // Novo parâmetro stream
{
    TORCH_CHECK(targets_bf16.size() == sources_fp32.size(),
                "foreach_copy_stochastic: Number of target and source tensors must match.");

    // cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // REMOVIDO DAQUI
    uint64_t current_global_element_offset = 0;

    for (size_t i = 0; i < targets_bf16.size(); ++i) {
        const auto& target_t = targets_bf16[i];
        const auto& source_t = sources_fp32[i];

        TORCH_CHECK(target_t.scalar_type() == at::kBFloat16, "Target tensor ", i, " must be BFloat16.");
        TORCH_CHECK(source_t.scalar_type() == at::kFloat, "Source tensor ", i, " must be Float32.");
        TORCH_CHECK(target_t.is_cuda() && source_t.is_cuda(), "Tensors ", i, " must be CUDA tensors.");
        TORCH_CHECK(target_t.sizes() == source_t.sizes(), "Target and source tensors ", i, " must have the same shape.");
        TORCH_CHECK(target_t.is_contiguous() && source_t.is_contiguous(),
                    "Tensors ", i, " must be contiguous. Call .contiguous() before passing.");

        if (target_t.numel() == 0) {
            continue;
        }

        at::BFloat16* target_ptr = target_t.data_ptr<at::BFloat16>();
        const float* source_ptr = source_t.data_ptr<float>();
        int64_t num_elements = target_t.numel();

        const int threads_per_block = 256;
        const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

        stochastic_copy_kernel<<<num_blocks, threads_per_block, 0, stream>>>( // Usa a stream passada
            target_ptr,
            source_ptr,
            num_elements,
            base_seed,
            current_global_element_offset
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed for tensor ", i, ": ", cudaGetErrorString(err));

        current_global_element_offset += num_elements;
    }
}