#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// <<< START: declaração do kernel CUDA >>>
extern "C" __global__ void prodigy_stats_kernel_f32(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    const float* __restrict__ param0,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ s,
    double* __restrict__ d_out,         // d_out[0] = d_num, d_out[1] = d_den
    float beta1, float beta2, float beta3,
    float d,   float d0, float dlr_for_dnum,
    float growth_rate, // Mantido aqui, mas verificar se é usado no kernel
    int64_t slice_p,
    int64_t numel
);
// <<< END: declaração do kernel CUDA >>>

// <<< START: launcher C++ que chama o kernel >>>
void prodigy_stats_cuda_launcher(
    torch::Tensor grad,
    torch::Tensor param,
    torch::Tensor param0,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor s,
    torch::Tensor d_out,            // tensor [2] float64
    float beta1, float beta2, float beta3,
    float d, float d0,
    float dlr_for_dnum, float growth_rate,
    int64_t slice_p
) {
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(param.is_cuda(), "param must be a CUDA tensor");
    TORCH_CHECK(param0.is_cuda(), "param0 must be a CUDA tensor");
    TORCH_CHECK(exp_avg.is_cuda(), "exp_avg must be a CUDA tensor");
    TORCH_CHECK(exp_avg_sq.is_cuda(), "exp_avg_sq must be a CUDA tensor");
    TORCH_CHECK(s.is_cuda(), "s must be a CUDA tensor");
    TORCH_CHECK(d_out.is_cuda(), "d_out must be a CUDA tensor");

    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(param.is_contiguous(), "param must be contiguous");
    // ... (adicione verificações de contiguidade para todos os tensores de entrada/saída)
    TORCH_CHECK(d_out.is_contiguous(), "d_out must be contiguous");


    TORCH_CHECK(d_out.numel() == 2, "d_out tensor must have 2 elements");
    TORCH_CHECK(d_out.scalar_type() == torch::kFloat64, "d_out tensor must be float64");


    int64_t n = param.numel();
    const int threads = 256; // Ou TUNED_THREADS se você tiver um valor ótimo
    const int blocks  = int((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    prodigy_stats_kernel_f32<<<blocks, threads, 0, stream>>>(
        grad.data_ptr<float>(),
        param.data_ptr<float>(),
        param0.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        s.data_ptr<float>(),
        d_out.data_ptr<double>(), // Passa o ponteiro para o tensor d_out
        beta1, beta2, beta3,
        d, d0, dlr_for_dnum,
        growth_rate, // Passando growth_rate
        slice_p,
        n
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "prodigy_stats_kernel_f32 launch failed: ",
                cudaGetErrorString(err));
}
// <<< END: launcher C++ que chama o kernel >>>

// <<< START: binding PyBind11 >>>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prodigy_stats_cuda",
          &prodigy_stats_cuda_launcher,
          "CUDA kernel para estatísticas do Prodigy (EMA + d_num/d_den)",
          py::arg("grad"),
          py::arg("param"),
          py::arg("param0"),
          py::arg("exp_avg"),
          py::arg("exp_avg_sq"),
          py::arg("s"),
          py::arg("d_out"),
          py::arg("beta1"),
          py::arg("beta2"),
          py::arg("beta3"),
          py::arg("d"),
          py::arg("d0"),
          py::arg("dlr_for_dnum"),
          py::arg("growth_rate"),
          py::arg("slice_p")
    );
}
// <<< END: binding PyBind11 >>>