// Arquivo: stochastic_copy_cuda.cu

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

// START: PRNG leve inline (LCG), sem alocar estado por thread
__device__ __forceinline__ uint32_t lcg32(uint64_t state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return static_cast<uint32_t>(state * 2685821657736338717ULL);
}
// END

// START: kernel fusion para TODOS os elementos de TODOS os tensores
__global__ void fused_stochastic_copy(
    at::BFloat16**    target_ptrs,
    const float**     source_ptrs,
    const int64_t*    offsets,
    int64_t           total_elems,
    int64_t           n_tensors,
    uint64_t          seed)
{
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elems) return;

    // determinar de que tensor e índice local se trata
    int tid = 0;
    // START: busca linear simples (n_tensors geralmente pequeno)
    for (int i = 0; i < n_tensors; ++i) {
        if (offsets[i+1] > gid) { tid = i; break; }
    }
    // END
    int64_t local_idx = gid - offsets[tid];

    // gerar 16 bits pseudo‐aleatórios
    uint32_t rnd32 = lcg32(seed ^ gid);
    uint16_t rnd16 = static_cast<uint16_t>(rnd32);

    // ler FP32 e aplicar máscara para BF16 estocástico
    float src = source_ptrs[tid][local_idx];
    int32_t src_bits = __float_as_int(src);
    int32_t tmp_bits = (rnd16 + src_bits) & 0xFFFF0000;
    float rounded = __int_as_float(tmp_bits);

    target_ptrs[tid][local_idx] = at::BFloat16(rounded);
}
// END

void foreach_copy_stochastic_cuda_launcher(
    const std::vector<at::Tensor>& targets_bf16,
    const std::vector<at::Tensor>& sources_fp32,
    uint64_t base_seed,
    cudaStream_t stream)
{
    TORCH_CHECK(targets_bf16.size() == sources_fp32.size(),
                "Número de tensores alvo e fonte deve ser igual.");

    int64_t n = (int64_t)targets_bf16.size();

    // START: coletar ponteiros e offsets em arrays host
    std::vector<at::BFloat16*> target_ptrs(n);
    std::vector<const float*> source_ptrs(n);
    std::vector<int64_t> offsets(n+1);
    offsets[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        const auto& T = targets_bf16[i];
        const auto& S = sources_fp32[i];
        TORCH_CHECK(T.is_cuda() && S.is_cuda(),   "Somente CUDA tensors.");
        TORCH_CHECK(T.scalar_type() == at::kBFloat16, "Target deve ser BF16.");
        TORCH_CHECK(S.scalar_type() == at::kFloat,    "Source deve ser Float32.");
        TORCH_CHECK(T.sizes() == S.sizes(),           "Mesma forma.");
        TORCH_CHECK(T.is_contiguous() && S.is_contiguous(),
                    "Chame .contiguous().");

        int64_t ne = T.numel();
        offsets[i+1]     = offsets[i] + ne;
        target_ptrs[i]   = T.data_ptr<at::BFloat16>();
        source_ptrs[i]   = S.data_ptr<float>();
    }
    int64_t total_elems = offsets[n];
    // END

    // START: alocar e copiar arrays de ponteiros e offsets para device
    at::BFloat16** d_target_ptrs = nullptr;
    const float** d_source_ptrs = nullptr;
    int64_t*      d_offsets     = nullptr;

    cudaMalloc(&d_target_ptrs, n * sizeof(at::BFloat16*));
    cudaMalloc(&d_source_ptrs, n * sizeof(const float*));
    cudaMalloc(&d_offsets,     (n+1) * sizeof(int64_t));

    cudaMemcpyAsync(
        d_target_ptrs, target_ptrs.data(),
        n * sizeof(at::BFloat16*),
        cudaMemcpyHostToDevice, stream
    );
    cudaMemcpyAsync(
        d_source_ptrs, source_ptrs.data(),
        n * sizeof(const float*),
        cudaMemcpyHostToDevice, stream
    );
    cudaMemcpyAsync(
        d_offsets, offsets.data(),
        (n+1) * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream
    );
    // END

    // START: único lançamento de kernel para todo o batch
    const int threads = 256;
    const int blocks  = (int64_t)((total_elems + threads - 1) / threads);
    fused_stochastic_copy<<<blocks, threads, 0, stream>>>(
        d_target_ptrs,
        d_source_ptrs,
        d_offsets,
        total_elems,
        n,
        base_seed
    );

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel fused falhou: ", cudaGetErrorString(err));
    // END

    // START: liberar buffers de ponteiros no device
    cudaFree(d_target_ptrs);
    cudaFree(d_source_ptrs);
    cudaFree(d_offsets);
    // END
}
