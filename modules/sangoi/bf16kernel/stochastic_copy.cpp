#include <torch/extension.h>
#include <vector>
#include <atomic>
#include <ATen/cuda/CUDAContext.h> // Para at::cuda::getCurrentCUDAStream()

// Declaração da função CUDA definida em stochastic_copy_cuda.cu - MODIFICADA
void foreach_copy_stochastic_cuda_launcher(
    const std::vector<at::Tensor>& targets_bf16,
    const std::vector<at::Tensor>& sources_fp32,
    uint64_t base_seed,
    cudaStream_t stream); // Novo parâmetro stream

// Função wrapper que será exposta ao Python via Pybind11 - MODIFICADA
void foreach_copy_stochastic_pybind_wrapper(
    const std::vector<at::Tensor>& targets_bf16,
    const std::vector<at::Tensor>& sources_fp32)
{
    static std::atomic<uint64_t> invocation_count = {0};
    uint64_t seed_for_this_call = 42ULL + invocation_count.fetch_add(1);

    // Obter a stream CUDA atual AQUI
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Passar a stream para a função launcher
    foreach_copy_stochastic_cuda_launcher(targets_bf16, sources_fp32, seed_for_this_call, stream);
}

// Ligação Pybind11 para expor a função ao Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foreach_copy_stochastic",
          &foreach_copy_stochastic_pybind_wrapper,
          "Performs stochastic copy from a list of FP32 tensors to a list of BF16 tensors using a CUDA kernel. \
          Both lists must have the same number of tensors, and corresponding tensors must have the same shape and be contiguous CUDA tensors.",
          py::arg("targets_bf16_list"),
          py::arg("sources_fp32_list")
    );
}