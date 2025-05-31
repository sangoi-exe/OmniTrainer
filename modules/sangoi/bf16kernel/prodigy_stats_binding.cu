#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <algorithm>
#include <vector>
#include <cuda_bf16.h> // Para __nv_bfloat16 e conversões

namespace py = pybind11;

// ============================================================================
// CONSTANTES E CONFIGURAÇÕES OTIMIZADAS
// ============================================================================
constexpr int DEFAULT_THREADS_PER_BLOCK = 256;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int MAX_BLOCKS = 65535; // Limite prático para número de blocos
constexpr int WARP_SIZE = 32;
constexpr size_t PREFETCH_THRESHOLD = 1024 * 1024;  // 1MB threshold para prefetch

// ============================================================================
// KERNEL CUDA OTIMIZADO
// ============================================================================
__global__ void prodigy_stats_kernel_bf16(
    const at::BFloat16* __restrict__ grad,
    const at::BFloat16* __restrict__ param,
    const at::BFloat16* __restrict__ param0,
    at::BFloat16* __restrict__ exp_avg,
    at::BFloat16* __restrict__ exp_avg_sq,
    at::BFloat16* __restrict__ s,
    double* __restrict__ d_out, // d_out[0] para d_num, d_out[1] para d_den
    float beta1, float beta2, float beta3,
    float d, float d0, float dlr_for_dnum, // d e d0 não são usados diretamente no kernel, mas dlr_for_dnum é
    float growth_rate, // Não usado atualmente no kernel, mas passado
    int64_t numel
) {
    // Cálculo otimizado do índice global
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Early return para threads fora do range
    if (idx >= numel) return;

    // ========================================================================
    // CARREGAMENTO COALESCED DE MEMÓRIA E CONVERSÃO PARA FLOAT
    // ========================================================================
    const float g_float = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&grad[idx]));
    const float p_float = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&param[idx]));
    const float p0_float = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&param0[idx]));
    const float ea_float = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&exp_avg[idx]));
    const float eas_float = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&exp_avg_sq[idx]));
    const float s_old_float = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s[idx]));

    // ========================================================================
    // CÁLCULOS MATEMÁTICOS OTIMIZADOS PARA EMA E S
    // ========================================================================

    // Pré-calcula constantes para evitar recomputação
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_beta3 = 1.0f - beta3;

    // Atualização das médias exponenciais (fusão de operações)
    const float exp_avg_new_float = fmaf(beta1, ea_float, one_minus_beta1 * g_float);
    const float g_squared_float = g_float * g_float;
    const float exp_avg_sq_new_float = fmaf(beta2, eas_float, one_minus_beta2 * g_squared_float);

    // Cálculo de s_k com operação fusionada
    const float param_diff_float = p_float - p0_float;
    const float abs_param_diff_float = fabsf(param_diff_float);
    const float s_new_float = fmaf(beta3, s_old_float, one_minus_beta3 * abs_param_diff_float);

    // ========================================================================
    // CÁLCULO E ACUMULAÇÃO ATÔMICA PARA d_num e d_den
    // ========================================================================
    // Converter para double para acumulação de maior precisão
    const double g_double = static_cast<double>(g_float);
    const double p_double = static_cast<double>(p_float);
    const double p0_double = static_cast<double>(p0_float);
    const double param_diff_double = p_double - p0_double;

    // Contribuição para d_num (numerador de d)
    const double d_num_contribution = static_cast<double>(dlr_for_dnum) * g_double * param_diff_double;

    // Contribuição para d_den (denominador de d)
    const double d_den_contribution = param_diff_double * param_diff_double;

    // Acumulação atômica em d_out (requer GPU com Compute Capability 6.x+ para double)
    atomicAdd(&d_out[0], d_num_contribution); // Acumula em d_num
    atomicAdd(&d_out[1], d_den_contribution); // Acumula em d_den

    // ========================================================================
    // ESCRITA COALESCED PARA MEMÓRIA GLOBAL (BF16)
    // ========================================================================
    *reinterpret_cast<__nv_bfloat16*>(&exp_avg[idx]) = __float2bfloat16(exp_avg_new_float);
    *reinterpret_cast<__nv_bfloat16*>(&exp_avg_sq[idx]) = __float2bfloat16(exp_avg_sq_new_float);
    *reinterpret_cast<__nv_bfloat16*>(&s[idx]) = __float2bfloat16(s_new_float);
}

// ============================================================================
// UTILITÁRIOS DE OTIMIZAÇÃO (getOptimalThreads não modificado)
// ============================================================================
int getOptimalThreads(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return DEFAULT_THREADS_PER_BLOCK;
    }
    int optimal = std::min(prop.maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
    return (optimal / WARP_SIZE) * WARP_SIZE;
}

// Função para obter configuração ótima (ainda usando Occupancy Calculator, pode ser um ponto de falha)
std::pair<int, int> getOptimalLaunchConfig(int device_id, int64_t numel) {
    int min_grid_size, block_size_occupancy;

    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size_occupancy,
        (void*)prodigy_stats_kernel_bf16, 0, 0
    );

    int final_block_size;
    if (err == cudaSuccess && block_size_occupancy > 0) {
        final_block_size = block_size_occupancy;
    } else {
        cudaGetLastError();
        final_block_size = getOptimalThreads(device_id);
    }
    
    final_block_size = std::max(WARP_SIZE, std::min(final_block_size, MAX_THREADS_PER_BLOCK));
    final_block_size = (final_block_size / WARP_SIZE) * WARP_SIZE;
    if (final_block_size == 0) final_block_size = WARP_SIZE;

    int blocks = (numel + final_block_size - 1) / final_block_size;
    blocks = std::min(blocks, MAX_BLOCKS);
    if (blocks == 0 && numel > 0) blocks = 1;

    return {blocks, final_block_size};
}

// ============================================================================
// LAUNCHER C++ OTIMIZADO
// ============================================================================
void prodigy_stats_cuda_launcher(
    torch::Tensor grad,
    torch::Tensor param,
    torch::Tensor param0,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor s,
    torch::Tensor d_out, // Tensor para [d_num, d_den]
    float beta1, float beta2, float beta3,
    float d, float d0,
    float dlr_for_dnum, float growth_rate
) {
    // ========================================================================
    // VERIFICAÇÕES ROBUSTAS DE ENTRADA
    // ========================================================================
    std::vector<torch::Tensor> input_tensors_bf16 = {grad, param, param0, exp_avg, exp_avg_sq, s};
    std::vector<std::string> tensor_names_bf16 = {"grad", "param", "param0", "exp_avg", "exp_avg_sq", "s"};

    for (size_t i = 0; i < input_tensors_bf16.size(); ++i) {
        TORCH_CHECK(input_tensors_bf16[i].is_cuda(), tensor_names_bf16[i], " must be a CUDA tensor");
        TORCH_CHECK(input_tensors_bf16[i].is_contiguous(), tensor_names_bf16[i], " must be contiguous");
        TORCH_CHECK(input_tensors_bf16[i].scalar_type() == torch::kBFloat16, tensor_names_bf16[i], " must be bfloat16");
    }

    TORCH_CHECK(d_out.is_cuda(), "d_out must be a CUDA tensor");
    TORCH_CHECK(d_out.is_contiguous(), "d_out must be contiguous");
    TORCH_CHECK(d_out.scalar_type() == torch::kFloat64, "d_out must be float64 for precision");
    TORCH_CHECK(d_out.numel() == 2, "d_out must have exactly 2 elements [d_num, d_den]");

    // NOTA: As checagens de shape foram movidas para o lado Python,
    // pois o Python está concatenando tudo em um único tensor 1D.
    // O kernel espera esses flat tensors já com shapes compatíveis.
    // No entanto, é bom garantir que o numel seja > 0.
    const int64_t numel = param.numel();
    TORCH_CHECK(numel > 0, "Tensors cannot be empty (numel > 0)");

    TORCH_CHECK(beta1 >= 0.0f && beta1 <= 1.0f, "beta1 must be in [0,1]");
    TORCH_CHECK(beta2 >= 0.0f && beta2 <= 1.0f, "beta2 must be in [0,1]");
    TORCH_CHECK(beta3 >= 0.0f && beta3 <= 1.0f, "beta3 must be in [0,1]");
    TORCH_CHECK(d > 0.0f, "d must be positive"); // d e d0 são usados no lado Python

    // ========================================================================
    // CONFIGURAÇÃO OTIMIZADA DE DEVICE E STREAM
    // ========================================================================
    const int device_idx = grad.device().index();
    cudaSetDevice(device_idx); // Garante que o contexto CUDA está correto para o device_idx
    const at::cuda::CUDAStream& current_stream = at::cuda::getCurrentCUDAStream(device_idx);
    cudaStream_t stream = current_stream.stream();

    // ========================================================================
    // MEMORY PREFETCHING PARA TENSORES GRANDES
    // ========================================================================
    const size_t total_bytes_main = grad.nbytes() + param.nbytes() + param0.nbytes() +
                                   exp_avg.nbytes() + exp_avg_sq.nbytes() + s.nbytes();

    if (total_bytes_main > PREFETCH_THRESHOLD) {
        for (const auto& tensor : {grad, param, param0, exp_avg, exp_avg_sq, s}) {
            if (tensor.nbytes() > 0) {
                 cudaMemPrefetchAsync(tensor.data_ptr(), tensor.nbytes(), device_idx, stream);
            }
        }
    }

    // ========================================================================
    // CONFIGURAÇÃO ÓTIMA DE LAUNCH (AJUSTADO PARA DEBUG)
    // ========================================================================
    // A função getOptimalLaunchConfig é boa, mas em alguns casos pode gerar um erro "invalid argument".
    // Para depuração, estamos temporariamente usando uma configuração fixa e robusta.
    // auto [blocks, threads] = getOptimalLaunchConfig(device_idx, numel); // Comentado para usar fixed values

    int blocks;
    int threads;

    // Define uma configuração fixa e segura para depuração
    // Ajuste 'fixed_threads' (e.g., 256, 512, 1024) baseado na sua GPU, deve ser múltiplo de WARP_SIZE (32)
    int fixed_threads_count = 256; 
    int calculated_blocks = (numel + fixed_threads_count - 1) / fixed_threads_count;
    
    blocks = std::min(calculated_blocks, MAX_BLOCKS);
    if (blocks == 0 && numel > 0) blocks = 1; // Garante pelo menos um bloco se houver elementos

    threads = fixed_threads_count; // Usar o valor fixo

    #ifdef DEBUG_KERNEL_LAUNCH
    std::cout << "[DEBUG] Kernel launch config: blocks=" << blocks 
              << ", threads=" << threads 
              << ", numel=" << numel 
              << ", device_idx=" << device_idx << std::endl;
    #endif

    // ========================================================================
    // INICIALIZAÇÃO DE d_out
    // ========================================================================
    cudaError_t memset_err = cudaMemsetAsync(d_out.data_ptr(), 0, d_out.nbytes(), stream);
    TORCH_CHECK(memset_err == cudaSuccess, "cudaMemsetAsync for d_out failed: ", cudaGetErrorString(memset_err));

    // ========================================================================
    // LAUNCH DO KERNEL OTIMIZADO
    // ========================================================================
    prodigy_stats_kernel_bf16<<<blocks, threads, 0, stream>>>(
        grad.data_ptr<at::BFloat16>(),
        param.data_ptr<at::BFloat16>(),
        param0.data_ptr<at::BFloat16>(),
        exp_avg.data_ptr<at::BFloat16>(),
        exp_avg_sq.data_ptr<at::BFloat16>(),
        s.data_ptr<at::BFloat16>(),
        d_out.data_ptr<double>(),
        beta1, beta2, beta3,
        d, d0, dlr_for_dnum, // Usando dlr_for_d_num_calc_step diretamente conforme o kernel espera
        growth_rate,
        numel
    );

    // ========================================================================
    // VERIFICAÇÃO DE ERRO APRIMORADA
    // ========================================================================
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "Kernel launch failed: ", cudaGetErrorString(launch_err),
                " (device=", device_idx,
                ", blocks=", blocks,
                ", threads=", threads,
                ", numel=", numel, ")");

    #ifdef DEBUG_KERNEL_SYNC
    cudaStreamSynchronize(stream);
    const cudaError_t exec_err = cudaGetLastError();
    TORCH_CHECK(exec_err == cudaSuccess,
                "Kernel execution failed after sync: ", cudaGetErrorString(exec_err));
    #endif
}


// ============================================================================
// LAUNCHER PARA MÚLTIPLOS GRUPOS DE PARÂMETROS (LORA FRIENDLY)  
// (Aplicando o mesmo fix de launch config para consistência)
// ============================================================================
void prodigy_stats_cuda_foreach_launcher(
    const std::vector<torch::Tensor>& grads,
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& params0,
    const std::vector<torch::Tensor>& exp_avgs,
    const std::vector<torch::Tensor>& exp_avg_sqs,
    const std::vector<torch::Tensor>& ss,
    torch::Tensor d_out,  // Compartilhado para todos os grupos
    float beta1, float beta2, float beta3,
    float d, float d0,
    float dlr_for_dnum, float growth_rate
) {
    TORCH_CHECK(!grads.empty(), "Input tensor lists cannot be empty for foreach launcher.");
    const size_t n_groups = grads.size();
    
    TORCH_CHECK(params.size() == n_groups, "Size mismatch: params list size must equal grads list size.");
    TORCH_CHECK(params0.size() == n_groups, "Size mismatch: params0 list size must equal grads list size.");
    TORCH_CHECK(exp_avgs.size() == n_groups, "Size mismatch: exp_avgs list size must equal grads list size.");
    TORCH_CHECK(exp_avg_sqs.size() == n_groups, "Size mismatch: exp_avg_sqs list size must equal grads list size.");
    TORCH_CHECK(ss.size() == n_groups, "Size mismatch: ss list size must equal grads list size.");

    TORCH_CHECK(d_out.is_cuda(), "d_out must be a CUDA tensor");
    TORCH_CHECK(d_out.is_contiguous(), "d_out must be contiguous");
    TORCH_CHECK(d_out.scalar_type() == torch::kFloat64, "d_out must be float64 for precision");
    TORCH_CHECK(d_out.numel() == 2, "d_out must have exactly 2 elements [d_num, d_den]");

    const int device_idx = grads[0].device().index(); // Assume todos os tensores no mesmo device
    cudaSetDevice(device_idx);
    const at::cuda::CUDAStream& current_stream = at::cuda::getCurrentCUDAStream(device_idx);
    cudaStream_t stream = current_stream.stream();
    
    // Zera d_out uma vez para todos os grupos
    cudaError_t memset_err = cudaMemsetAsync(d_out.data_ptr(), 0, d_out.nbytes(), stream);
    TORCH_CHECK(memset_err == cudaSuccess, "cudaMemsetAsync for d_out in foreach failed: ", cudaGetErrorString(memset_err));
    
    // Processa cada grupo independentemente
    for (size_t i = 0; i < n_groups; ++i) {
        const torch::Tensor& grad_i = grads[i];
        const torch::Tensor& param_i = params[i];
        const torch::Tensor& param0_i = params0[i];
        const torch::Tensor& exp_avg_i = exp_avgs[i];
        const torch::Tensor& exp_avg_sq_i = exp_avg_sqs[i];
        const torch::Tensor& s_i = ss[i];
        
        TORCH_CHECK(grad_i.is_cuda() && param_i.is_cuda() && param0_i.is_cuda() &&
                    exp_avg_i.is_cuda() && exp_avg_sq_i.is_cuda() && s_i.is_cuda(),
                    "All tensors for group ", i, " must be CUDA tensors.");
        TORCH_CHECK(grad_i.scalar_type() == torch::kBFloat16 && param_i.scalar_type() == torch::kBFloat16 &&
                    param0_i.scalar_type() == torch::kBFloat16 && exp_avg_i.scalar_type() == torch::kBFloat16 &&
                    exp_avg_sq_i.scalar_type() == torch::kBFloat16 && s_i.scalar_type() == torch::kBFloat16,
                    "All tensors for group ", i, " must be bfloat16.");
        TORCH_CHECK(grad_i.device() == grads[0].device(), "All tensors must be on the same device.");

        // O shape check já está feito no Python com os .flatten()
        // O importante é garantir que o numel seja > 0 para lançar o kernel
        const int64_t numel = param_i.numel();
        if (numel == 0) continue; 

        // Configuração ótima para este grupo (aplicando o mesmo fix de launch config)
        // auto [blocks, threads] = getOptimalLaunchConfig(device_idx, numel); // Comentado

        int blocks;
        int threads;
        int fixed_threads_count = 256; 
        int calculated_blocks = (numel + fixed_threads_count - 1) / fixed_threads_count;
        
        blocks = std::min(calculated_blocks, MAX_BLOCKS);
        if (blocks == 0 && numel > 0) blocks = 1;

        threads = fixed_threads_count;

        // Launch do kernel para este grupo
        prodigy_stats_kernel_bf16<<<blocks, threads, 0, stream>>>(
            grad_i.data_ptr<at::BFloat16>(),
            param_i.data_ptr<at::BFloat16>(),
            param0_i.data_ptr<at::BFloat16>(),
            exp_avg_i.data_ptr<at::BFloat16>(),
            exp_avg_sq_i.data_ptr<at::BFloat16>(),
            s_i.data_ptr<at::BFloat16>(),
            d_out.data_ptr<double>(),
            beta1, beta2, beta3,
            d, d0, dlr_for_dnum, // dlr_for_dnum é passado para cada chamada
            growth_rate,
            numel
        );
        
        const cudaError_t launch_err_group = cudaGetLastError();
        TORCH_CHECK(launch_err_group == cudaSuccess,
                    "Kernel launch failed for group ", i, ": ", 
                    cudaGetErrorString(launch_err_group),
                    " (device=", device_idx,
                    ", blocks=", blocks,
                    ", threads=", threads,
                    ", numel=", numel, ")");
    }
    #ifdef DEBUG_KERNEL_SYNC
    cudaStreamSynchronize(stream);
    const cudaError_t exec_err = cudaGetLastError();
    TORCH_CHECK(exec_err == cudaSuccess,
                "Foreach kernel execution failed after sync: ", cudaGetErrorString(exec_err));
    #endif
}


// ============================================================================
// FUNÇÃO AUXILIAR PARA BENCHMARKING (OPCIONAL, não modificada significativamente)
// ============================================================================
std::tuple<double, int, int> benchmark_kernel_config(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param0,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor s,
    torch::Tensor d_out,
    float beta1, float beta2, float beta3,
    float d, float d0,
    float dlr_for_dnum, float growth_rate,
    int warmup_runs = 5,
    int timing_runs = 10
) {
    const int device_idx = grad.device().index();
    cudaSetDevice(device_idx);
    
    // Usar a mesma lógica de configuração de lançamento temporária
    int blocks;
    int threads;
    int fixed_threads_count = 256; 
    int calculated_blocks = (param.numel() + fixed_threads_count - 1) / fixed_threads_count;
    
    blocks = std::min(calculated_blocks, MAX_BLOCKS);
    if (blocks == 0 && param.numel() > 0) blocks = 1;
    threads = fixed_threads_count;


    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_idx).stream();

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        cudaMemsetAsync(d_out.data_ptr(), 0, d_out.nbytes(), stream); // Zera d_out
        prodigy_stats_kernel_bf16<<<blocks, threads, 0, stream>>>(
            grad.data_ptr<at::BFloat16>(), param.data_ptr<at::BFloat16>(), param0.data_ptr<at::BFloat16>(),
            exp_avg.data_ptr<at::BFloat16>(), exp_avg_sq.data_ptr<at::BFloat16>(), s.data_ptr<at::BFloat16>(),
            d_out.data_ptr<double>(), beta1, beta2, beta3, d, d0, dlr_for_dnum,
            growth_rate, param.numel()
        );
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Timing
    for (int i = 0; i < timing_runs; ++i) {
        cudaMemsetAsync(d_out.data_ptr(), 0, d_out.nbytes(), stream); // Zera d_out
        prodigy_stats_kernel_bf16<<<blocks, threads, 0, stream>>>(
            grad.data_ptr<at::BFloat16>(), param.data_ptr<at::BFloat16>(), param0.data_ptr<at::BFloat16>(),
            exp_avg.data_ptr<at::BFloat16>(), exp_avg_sq.data_ptr<at::BFloat16>(), s.data_ptr<at::BFloat16>(),
            d_out.data_ptr<double>(), beta1, beta2, beta3, d, d0, dlr_for_dnum,
            growth_rate, param.numel()
        );
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_tuple(static_cast<double>(elapsed_ms) / timing_runs, blocks, threads);
}

// ============================================================================
// BINDING PYBIND11 APRIMORADO (não modificado)
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        Otimizador Prodigy CUDA Extension
        
        Implementação CUDA otimizada para cálculos de estatísticas do otimizador Prodigy.
        Inclui técnicas avançadas de otimização como memory prefetching, 
        configuração dinâmica de threads, e operações atômicas thread-safe.
    )pbdoc";

    m.def("prodigy_stats_cuda",
          &prodigy_stats_cuda_launcher,
          R"pbdoc(
          Kernel CUDA otimizado para estatísticas do otimizador Prodigy.
          
          Calcula médias exponenciais (EMA), estatística s, e contribuições para d_num/d_den
          com otimizações avançadas de performance.
          
          Args:
              grad (Tensor): Gradientes (bfloat16, CUDA, contiguous)
              param (Tensor): Parâmetros atuais (bfloat16, CUDA, contiguous)
              param0 (Tensor): Parâmetros iniciais (bfloat16, CUDA, contiguous)
              exp_avg (Tensor): Média exponencial dos gradientes (bfloat16, CUDA, contiguous)
              exp_avg_sq (Tensor): Média exponencial dos gradientes² (bfloat16, CUDA, contiguous)
              s (Tensor): Estatística de diferença de parâmetros (bfloat16, CUDA, contiguous)
              d_out (Tensor): Tensor de saída [d_num, d_den] (float64, CUDA, contiguous, size 2)
              beta1 (float): Fator de decaimento para exp_avg [0,1]
              beta2 (float): Fator de decaimento para exp_avg_sq [0,1]
              beta3 (float): Fator de decaimento para s [0,1]
              d (float): Parâmetro d do Prodigy (>0) (Usado no Python, passado para consistência)
              d0 (float): Parâmetro d0 do Prodigy (Usado no Python, passado para consistência)
              dlr_for_dnum (float): Coeficiente para o cálculo de d_num.
              growth_rate (float): Taxa de crescimento (Usado no Python, passado para consistência)
              
          Returns:
              None: Operação in-place nos tensores exp_avg, exp_avg_sq, s e d_out.
          )pbdoc",
          py::arg("grad"), py::arg("param"), py::arg("param0"),
          py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("s"),
          py::arg("d_out"),
          py::arg("beta1"), py::arg("beta2"), py::arg("beta3"),
          py::arg("d"), py::arg("d0"),
          py::arg("dlr_for_dnum"), py::arg("growth_rate")
    );

    m.def("prodigy_stats_cuda_foreach",
          &prodigy_stats_cuda_foreach_launcher,
          "Kernel CUDA para múltiplos grupos de parâmetros (ideal para LoRA)",
          py::arg("grads"), py::arg("params"), py::arg("params0"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"), py::arg("ss"), py::arg("d_out"),
          py::arg("beta1"), py::arg("beta2"), py::arg("beta3"),
          py::arg("d"), py::arg("d0"), py::arg("dlr_for_dnum"), py::arg("growth_rate"));

    m.def("benchmark_kernel",
          &benchmark_kernel_config,
          R"pbdoc(
          Benchmarking function para medir performance do kernel.
          )pbdoc",
          py::arg("grad"), py::arg("param"), py::arg("param0"),
          py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("s"), py::arg("d_out"),
          py::arg("beta1"), py::arg("beta2"), py::arg("beta3"),
          py::arg("d"), py::arg("d0"), py::arg("dlr_for_dnum"), py::arg("growth_rate"),
          py::arg("warmup_runs") = 5, py::arg("timing_runs") = 10
    );
    
    m.def("get_optimal_threads", &getOptimalThreads,
          "Retorna número ótimo de threads para o device",
          py::arg("device_id"));
}