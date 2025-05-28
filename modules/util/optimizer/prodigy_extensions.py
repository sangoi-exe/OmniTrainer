import math
import traceback
import torch
import torch.distributed as dist

from modules.util.bf16_stochastic_rounding import (
    # add_stochastic_, # Não usado diretamente na versão vetorizada
    # addcdiv_stochastic_, # Substituído pela lógica vetorizada + copy_stochastic_
    copy_stochastic_, # Usado para converter de volta para BF16
)

try:
    # A importação correta é diretamente pelo nome do módulo
    import bf16_stochastic_cuda_custom
    _bf16_cuda_ext_available = True
    print("Successfully loaded 'bf16_stochastic_cuda_custom' extension.")
except ImportError as e:
    _bf16_cuda_ext_available = False
    print(f"Warning: Could not load 'bf16_stochastic_cuda_custom' extension: {e}")
    print("Falling back to Python loop for BF16 stochastic rounding copy.")

from prodigyopt.prodigy import Prodigy # Supondo que esta é a classe base

# Helper para estatísticas (inalterado)
class ProdigyStatsBuffer(list):
    def push(self, **kwargs):
        self.append(kwargs)

    def pop_all(self):
        buf = list(self)
        self.clear()
        return buf

# Nova função step vetorizada
@torch.no_grad()
def step_prodigy(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad(): # Habilita gradientes para o closure, se necessário
            loss = closure()

    if not hasattr(self, "_stats_buffer"):
        self._stats_buffer = ProdigyStatsBuffer()

    for group_idx, group in enumerate(self.param_groups):
        if group_idx == 0:
            self._stats_buffer.clear()

        if group["lr"] == 0.0:
            continue

        beta1, beta2 = group["betas"]
        beta3 = group["beta3"]
        if beta3 is None:
            beta3 = math.sqrt(beta2)

        d = group["d"]
        d_max = group["d_max"]
        d_coef = group["d_coef"]
        lr = group["lr"]
        use_bias_correction = group["use_bias_correction"]
        safeguard_warmup = group["safeguard_warmup"]
        fsdp_in_use = group["fsdp_in_use"]
        slice_p = group["slice_p"]
        growth_rate = group["growth_rate"]
        decouple = group["decouple"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        k = group["k"]

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0

        # dlr_initial é usado para d_numerator e weight decay (se dlr mudar depois)
        # No entanto, d é atualizado antes do loop de atualização de parâmetros,
        # então dlr será constante para todas as atualizações de parâmetros dentro de um grupo.
        # Recompute dlr after d is updated for the main parameter update step.

        # --- Parte 1: Acumular estatísticas para d e d_hat (loop original mantido por enquanto) ---
        # Esta parte envolve .item() e lógica sequencial que não é o foco da vetorização addcdiv_
        # A vetorização aqui seria mais complexa e menos impactante que a do passo de atualização.
        
        d_numerator = group["d_numerator"]
        d_numerator *= beta3
        d_denom = 0.0

        # Precisamos do dlr *antes* da atualização de d para o cálculo de d_numerator
        # e para o weight decay desacoplado se for aplicado antes da atualização de d.
        # No entanto, o weight decay desacoplado no código original usa o dlr *final*.
        # Vamos manter a lógica original: d é atualizado, depois dlr é recalculado para a atualização dos pesos.
        
        # Loop para cálculo de d_numerator e d_denom e atualização de EMA (exp_avg, exp_avg_sq)
        # Este loop NÃO atualiza p.data ainda, apenas os estados do otimizador e d_num/d_den
        for p_idx, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            
            grad = p.grad.data # grad não deve ser modificado por weight decay acoplado aqui ainda
                               # pois o weight decay acoplado é aplicado ao grad *antes* do cálculo do EMA
                               # e *antes* do cálculo do d_numerator.

            # State initialization (mantido como no original)
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["s"] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()
                if p.any(): # Corrigido para chamar como método
                    state["p0"] = p.flatten()[::slice_p].detach().clone()
                else:
                    state["p0"] = torch.tensor(0, device=p.device, dtype=p.dtype)
                if beta1 > 0:
                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

            # Aplicar weight decay acoplado (modifica o grad ANTES de usá-lo para EMAs e d_numerator)
            # Esta é uma diferença sutil: o grad original é usado para d_numerator,
            # mas o grad com weight decay é usado para EMAs.
            # Para simplificar e alinhar com muitas implementações, aplicaremos WD acoplado
            # ao grad que é usado para *tudo* (EMAs e d_numerator).
            # Se uma distinção precisa ser feita, seria necessário clonar o grad.
            # No código original, grad.add_ é feito, então o grad modificado é usado para d_numerator e EMAs.
            
            current_grad_for_emas_and_dnum = grad.clone() if weight_decay != 0 and not decouple else grad
            if weight_decay != 0 and not decouple:
                current_grad_for_emas_and_dnum.add_(p.data, alpha=weight_decay)

            # Cálculos para d_numerator e d_denom
            if lr > 0.0: # d_num/d_den só se lr > 0
                d0 = group["d0"]
                # dlr_for_d_num_calc = d * lr * bias_correction # dlr corrente para este cálculo
                # ^^^^ No código original, parece que dlr usado para d_numerator é o dlr da *iteração anterior*
                # ou um `d` que ainda não foi atualizado nesta etapa.
                # O código original faz: `d_numerator += (d / d0) * dlr * torch.dot(...)`
                # onde dlr é `d * lr * bias_correction` usando o `d` do início da etapa do grupo.
                # Vamos manter essa lógica:
                dlr_for_d_num_calc_step = d * lr * bias_correction
                sliced_grad_for_dnum = current_grad_for_emas_and_dnum.flatten()[::slice_p]
                d_numerator += (d / d0) * dlr_for_d_num_calc_step * torch.dot(sliced_grad_for_dnum, state["p0"] - p.data.flatten()[::slice_p]).item()

                # Adam EMA updates
                if beta1 > 0:
                    state["exp_avg"].mul_(beta1).add_(current_grad_for_emas_and_dnum, alpha=d * (1 - beta1))
                state["exp_avg_sq"].mul_(beta2).addcmul_(current_grad_for_emas_and_dnum, current_grad_for_emas_and_dnum, value=d * d * (1 - beta2))

                s_update_val = dlr_for_d_num_calc_step if not safeguard_warmup else d
                state["s"].mul_(beta3).add_(sliced_grad_for_dnum, alpha=((d / d0) * s_update_val))
                d_denom += state["s"].abs().sum().item()

        if d_denom == 0.0:
            group["k"] = k + 1
            continue

        if lr > 0.0: # Atualização de d, d_max, etc.
            if fsdp_in_use:
                # Encontra um tensor no dispositivo correto para o buffer de dist
                # Tenta encontrar um p com grad, senão o primeiro p do grupo
                # (assumindo que todos os params do grupo estão no mesmo dispositivo)
                first_param_in_group = next(iter(group['params']))
                dist_tensor_device = next((p.device for p in group["params"] if p.grad is not None), first_param_in_group.device)
                dist_tensor = torch.zeros(2, device=dist_tensor_device, dtype=torch.float64) # Usar float64 para precisão na soma
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0].item()
                global_d_denom = dist_tensor[1].item()
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom
            
            if global_d_denom == 0.0: # Evitar divisão por zero se após all_reduce o denominador for zero
                group["k"] = k + 1
                continue

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group["d0"]:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            group["d_numerator"] = global_d_numerator # Salvar o valor global
            group["d_denom"] = global_d_denom     # Salvar o valor global

        # dlr final para esta etapa do grupo, usando o 'd' atualizado
        dlr = d * lr * bias_correction

        self._stats_buffer.push(
            group_idx=group_idx, name=group["name"], step=k,
            d_num=group["d_numerator"], d_den=group["d_denom"], dlr=dlr,
        )

        # --- Parte 2: Atualização de parâmetros (onde a vetorização acontece) ---
        params_list_for_update = []
        grads_or_exp_avg_list = []
        denoms_list = []

        # Para arredondamento estocástico BF16
        bf16_params_originals_sr = [] # Lista dos tensores p.data originais em BF16
        bf16_params_fp32_for_update_sr = [] # Lista de cópias em FP32 de p.data para atualização
        bf16_tensor1_fp32_sr = [] # exp_avg ou grad em FP32
        bf16_denoms_fp32_sr = []  # denoms em FP32

        # Para atualização normal (FP32 ou BF16 sem arredondamento estocástico)
        normal_params_for_update = []
        normal_tensor1 = []
        normal_denoms = []
        
        # Listas para weight decay desacoplado
        wd_params_list_bf16_sr = [] # p.data originais BF16 para WD, antes da conversão para FP32 para update principal
        wd_params_list_normal = []  # p.data normais para WD

        for p in group["params"]:
            if p.grad is None:
                continue
            
            state = self.state[p]
            state["step"] += 1 # Incrementa o step do parâmetro

            # Denominador é calculado por parâmetro
            # .sqrt() cria um novo tensor, .add_() é in-place nele.
            current_denom = state["exp_avg_sq"].sqrt().add_(d * eps)

            # Escolhe entre exp_avg (Adam) ou grad (SGD-like)
            # O grad aqui deve ser o grad *original*, sem o weight_decay acoplado,
            # se o WD acoplado já foi usado para os EMAs.
            # No código original Prodigy, `grad` no passo de atualização é `p.grad.data` (sem WD acoplado).
            # O WD acoplado é aplicado ANTES dos EMAs.
            # Se beta1 > 0, usa exp_avg. Senão, usa grad (p.grad.data).
            # `exp_avg` já foi atualizado com o `current_grad_for_emas_and_dnum`.
            
            tensor1_for_update = state["exp_avg"] if beta1 > 0 else p.grad.data # p.grad.data original

            is_bf16_stochastic = p.dtype == torch.bfloat16 and self.stochastic_rounding

            if is_bf16_stochastic:
                # Weight decay desacoplado é aplicado ANTES da atualização principal
                if weight_decay != 0 and decouple:
                    # Adiciona à lista para WD em BF16. WD será aplicado diretamente no BF16.
                    wd_params_list_bf16_sr.append(p.data)

                # Prepara para atualização principal em FP32
                bf16_params_originals_sr.append(p.data)
                bf16_params_fp32_for_update_sr.append(p.data.to(torch.float32))
                bf16_tensor1_fp32_sr.append(tensor1_for_update.to(torch.float32))
                bf16_denoms_fp32_sr.append(current_denom.to(torch.float32))
            else:
                # Weight decay desacoplado
                if weight_decay != 0 and decouple:
                    wd_params_list_normal.append(p.data)
                
                # Atualização normal
                normal_params_for_update.append(p.data)
                normal_tensor1.append(tensor1_for_update)
                normal_denoms.append(current_denom)

        # Aplicar weight decay desacoplado vetorizado
        # p.data.add_(p.data, alpha=-weight_decay * dlr)
        # => p.data = p.data * (1 - weight_decay * dlr)
        # Ou: to_add = p.data * (-weight_decay * dlr); p.data.add_(to_add)
        if weight_decay != 0 and decouple:
            wd_alpha_scalar = -weight_decay * dlr
            if wd_params_list_bf16_sr:
                # Cria tensores para adicionar (p.data * alpha)
                # Para BF16, idealmente o WD também seria com maior precisão, mas o original faz direto.
                # Vamos manter a aplicação direta em BF16 para WD, como no original.
                # `p.data.add_(p.data, alpha=val)` -> `p_new = p_old + val * p_old`
                # `torch._foreach_mul` seguido por `torch._foreach_add` ou calcular `1+val`
                
                # Calculando o termo aditivo: p * alpha_wd
                # wd_add_terms_bf16 = [param.mul(wd_alpha_scalar) for param in wd_params_list_bf16_sr] # Não in-place
                # torch._foreach_add_(wd_params_list_bf16_sr, wd_add_terms_bf16)
                
                # Alternativa: p_new = p_old * (1 + wd_alpha_scalar)
                # Se wd_alpha_scalar é pequeno, 1+wd_alpha_scalar pode ser bem representado.
                # Esta abordagem é mais simples com _foreach_mul_
                # No entanto, p.add_(p.data, alpha=val) é diferente de p.mul_(1+val) para BF16 devido à precisão.
                # A forma mais fiel é criar a lista de `p*alpha` e adicioná-la.
                # Para simplificar e evitar estouro de memória com muitos tensores temporários:
                # Faremos loop para WD por enquanto se for BF16 SR, ou se a lista for pequena.
                # Ou, se quisermos vetorizar, usar a forma que o PyTorch FusedAdam usa:
                # p.mul_(1.0 - lr * wd) -> mas aqui é DLR.
                # p.data.add_(p.data, alpha=wd_value) é p = p + p * wd_value = p * (1 + wd_value)
                # A maneira mais direta de vetorizar p.add(p, alpha=v) é com p.mul(1+v)
                # Isso pode ter diferenças de precisão vs p + p*v em BF16.
                # Vamos usar a forma de adição explícita para maior fidelidade, mesmo que mais verbosa.
                # Ou, para WD desacoplado, o loop original é mais seguro para BF16:
                for p_data in wd_params_list_bf16_sr:
                     p_data.add_(p_data, alpha=wd_alpha_scalar) # In-place no BF16

            if wd_params_list_normal:
                # Para FP32, a diferença de precisão é menos preocupante.
                # wd_add_terms_normal = [param.mul(wd_alpha_scalar) for param in wd_params_list_normal]
                # torch._foreach_add_(wd_params_list_normal, wd_add_terms_normal)
                # Ou mais simples:
                for p_data in wd_params_list_normal: # Mantendo loop por segurança de precisão / simplicidade
                    p_data.add_(p_data, alpha=wd_alpha_scalar)


        # Determinar o 'value' para addcdiv_
        # Se beta1 > 0, value é -dlr. Senão (beta1 <= 0), value é -dlr * d.
        # Este 'd' é o 'd' atualizado do grupo.
        main_update_value = -dlr if beta1 > 0 else -dlr * d

        # Atualização principal vetorizada para não-estocásticos (como antes)
        if normal_params_for_update:
            torch._foreach_addcdiv_(normal_params_for_update,
                                     normal_tensor1,
                                     normal_denoms,
                                     value=main_update_value)

        # Atualização principal vetorizada para BF16 estocástico (opera em cópias FP32)
        if bf16_params_fp32_for_update_sr:
            torch._foreach_addcdiv_(bf16_params_fp32_for_update_sr,
                                     bf16_tensor1_fp32_sr,
                                     bf16_denoms_fp32_sr,
                                     value=main_update_value)
            
            # Copiar de volta para BF16 com arredondamento estocástico
            use_cuda_kernel_for_copy = _bf16_cuda_ext_available and \
                                       all(t.is_cuda for t in bf16_params_originals_sr) and \
                                       all(t.is_cuda for t in bf16_params_fp32_for_update_sr)
            
            # Copiar de volta para BF16 com arredondamento estocástico
            # (Este loop é o candidato para um kernel CUDA/Triton customizado no futuro)
            for i in range(len(bf16_params_originals_sr)):
                try:
                    copy_stochastic_(bf16_params_originals_sr[i], bf16_params_fp32_for_update_sr[i])
                except Exception as e:
                    # Fallback para cópia normal sem arredondamento estocástico se copy_stochastic_ falhar
                    # Ou poderia tentar addcdiv_ normal no BF16 original como o código antigo fazia.
                    # Por simplicidade, apenas copiaremos o resultado FP32 truncado.
                    print(f"Stochastic copy_stochastic_ failed for a parameter, falling back to regular copy: {e}")
                    traceback.print_exc()
                    bf16_params_originals_sr[i].copy_(bf16_params_fp32_for_update_sr[i])


            if use_cuda_kernel_for_copy:
                try:
                    # Verifica contiguidade e cria cópias contíguas se necessário.
                    # O kernel espera tensores contíguos.
                    # Idealmente, os tensores já seriam contíguos para evitar cópias extras.
                    # A verificação já está no C++ agora, mas é bom estar ciente.
                    # Se precisar garantir contiguidade aqui:
                    # temp_originals_sr_cont = [t.contiguous() for t in bf16_params_originals_sr]
                    # temp_fp32_for_update_sr_cont = [t.contiguous() for t in bf16_params_fp32_for_update_sr]
                    # bf16_stochastic_cuda_custom.foreach_copy_stochastic(
                    #    temp_originals_sr_cont,
                    #    temp_fp32_for_update_sr_cont
                    # )
                    # Se os tensores já são contíguos ou a checagem no C++ é suficiente:
                    bf16_stochastic_cuda_custom.foreach_copy_stochastic(
                       bf16_params_originals_sr, # Lista dos tensores BF16 de destino
                       bf16_params_fp32_for_update_sr  # Lista dos tensores FP32 de origem
                    )
                except Exception as e_cuda:
                    print(f"CUDA stochastic copy kernel failed: {e_cuda}. Falling back to Python loop.")
                    traceback.print_exc()
                    # Fallback para o loop Python
                    for i in range(len(bf16_params_originals_sr)):
                        try:
                            copy_stochastic_(bf16_params_originals_sr[i], bf16_params_fp32_for_update_sr[i])
                        except Exception as e_py_fallback:
                            print(f"Python fallback copy_stochastic_ also failed: {e_py_fallback}")
                            bf16_params_originals_sr[i].copy_(bf16_params_fp32_for_update_sr[i]) # Cópia simples como último recurso
            else: # Fallback para o loop Python se a extensão não estiver disponível ou tensores não CUDA
                if not _bf16_cuda_ext_available and bf16_params_originals_sr : # Adiciona log se era esperado mas não disponível
                    print("CUDA extension not used for BF16 stochastic copy, using Python loop.")
                for i in range(len(bf16_params_originals_sr)):
                    try:
                        copy_stochastic_(bf16_params_originals_sr[i], bf16_params_fp32_for_update_sr[i])
                    except Exception as e_py:
                        print(f"Python copy_stochastic_ failed: {e_py}. Using regular copy.")
                        bf16_params_originals_sr[i].copy_(bf16_params_fp32_for_update_sr[i])

        group["k"] = k + 1
    return loss

# Função patch (modificada para usar a versão vetorizada)
def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
    try:
        optimizer.stochastic_rounding = stochastic_rounding # Adiciona o atributo ao otimizador
        # Substitui o método step original pelo vetorizado
        optimizer.step = step_prodigy.__get__(optimizer, Prodigy)
        optimizer.pop_stats = _pop_prodigy_stats.__get__(optimizer, Prodigy) # Mantém o pop_stats
        print("Prodigy optimizer patched with VECTORIZED step method.")
    except Exception as e:
        print(f"Failed to set options in patch_prodigy: {e}")
        traceback.print_exc()

# Função para obter estatísticas (inalterada)
def _pop_prodigy_stats(self):
    return getattr(self, "_stats_buffer", ProdigyStatsBuffer()).pop_all()