// Arquivo: stochastic_copy.cpp

#include <torch/extension.h>
#include <vector>
#include <atomic>
#include <ATen/cuda/CUDAContext.h> // Para at::cuda::getCurrentCUDAStream()

// Declaração do launcher unificado
void foreach_copy_stochastic_cuda_launcher(
    const std::vector<at::Tensor>& targets_bf16,
    const std::vector<at::Tensor>& sources_fp32,
    uint64_t base_seed,
    cudaStream_t stream);

// Wrapper PyBind11
void foreach_copy_stochastic_pybind_wrapper(
    const std::vector<at::Tensor>& targets_bf16,
    const std::vector<at::Tensor>& sources_fp32)
{
    static std::atomic<uint64_t> invocation_count{0};
    uint64_t seed = 42ULL + invocation_count.fetch_add(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    foreach_copy_stochastic_cuda_launcher(targets_bf16, sources_fp32, seed, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foreach_copy_stochastic",
          &foreach_copy_stochastic_pybind_wrapper,
          "Stochastic BF16 copy (LCG+máscara) de lista de tensores FP32 para BF16.",
          py::arg("targets_bf16_list"),
          py::arg("sources_fp32_list"));
}
