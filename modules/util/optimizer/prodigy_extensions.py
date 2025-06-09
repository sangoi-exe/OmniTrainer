import math
import traceback
import torch
import torch.distributed as dist

from modules.util.bf16_stochastic_rounding import (
    add_stochastic_,
    addcdiv_stochastic_,
)

from prodigyopt.prodigy import Prodigy


@torch.no_grad()
def step_prodigy(self, closure=None):
    """Performs a single optimization step.

    Arguments:
            closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    if not hasattr(self, "_stats_buffer"):

        self._stats_buffer = ProdigyStatsBuffer()
    # limpa o buffer no primeiro grupo da step

    # iterate over each parameter group independently
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
        lr = group["lr"]  # now each group has its own LR
        use_bias_correction = group["use_bias_correction"]
        safeguard_warmup = group["safeguard_warmup"]
        fsdp_in_use = group["fsdp_in_use"]
        slice_p = group["slice_p"]
        growth_rate = group["growth_rate"]
        decouple = group["decouple"]

        # group's iteration counter (k)
        k = group["k"]

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0

        dlr = d * lr * bias_correction

        # we use the group's local d_numerator and locally reset the denominator
        d_numerator = group["d_numerator"]
        d_numerator *= beta3

        d_denom = 0.0  # group's denominator (will accumulate terms)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Apply weight decay (coupled variant)
            if group["weight_decay"] != 0 and not decouple:
                grad.add_(p.data, alpha=group["weight_decay"])

            # State initialization
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["s"] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                if p.any():
                    state["p0"] = p.flatten()[::slice_p].detach().clone()
                else:
                    # All values are zero, so save VRAM with a zero-tensor
                    state["p0"] = torch.tensor(0, device=p.device, dtype=p.dtype)

                # Exponential moving average of gradient values
                if beta1 > 0:
                    state["exp_avg"] = torch.zeros_like(p.data).detach()

                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

            exp_avg_sq = state["exp_avg_sq"]
            s = state["s"]
            p0 = state["p0"]

            # only if the group's LR is > 0 do we accumulate statistics
            if lr > 0.0:
                d0 = group["d0"]
                # we use d / d0 instead of just d to avoid getting values that are too small
                sliced_grad = grad.flatten()[::slice_p]
                d_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()

                # Adam EMA updates
                if beta1 > 0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))

                if safeguard_warmup:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                else:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))

                # accumulate the total denominator for this group
                d_denom += s.abs().sum().item()

        # if we didn't accumulate any grad (e.g., grad = 0), skip this group
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0.0:
            group["k"] = k + 1
            continue

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).to(next(p for p in group["params"] if p.grad is not None).device)
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            # 1. Gradiente médio do grupo (|g|) -> escalar
            # Usamos tensor para evitar problemas de device e permitir detach()
            device_param = next(p for p in group["params"] if p.grad is not None).device
            grad_abs_sum = torch.tensor(0.0, device=device_param)
            grad_elem_cnt = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                grad_abs_sum += g.abs().sum().item()
                grad_elem_cnt += g.numel()
            grad_mean = grad_abs_sum / (grad_elem_cnt + 1e-8)

            # 2. Estado por grupo para EMA
            dstate     = self.state.setdefault("_dcoef_state", {})
            ema_prev   = dstate.get(group_idx, grad_mean.detach())
            beta_dc    = group.get("dcoef_beta", 0.9)
            ema_curr   = beta_dc * ema_prev + (1 - beta_dc) * grad_mean
            dstate[group_idx] = ema_curr

            # 3. Novo d_coef
            target     = group.get("dcoef_target", 0.05)
            dc_min     = group.get("dcoef_min", 0.25)
            dc_max     = group.get("dcoef_max", 4.0)

            new_dcoef = torch.clamp(target / (ema_curr + 1e-8), dc_min, dc_max).item()
            group["d_coef"] = new_dcoef  # overwrite para o passo atual

            # compute d_hat and d_max for the current group
            d_hat = d_coef * global_d_numerator / global_d_denom

            # if it's the group's first update (d == d0), enforce d >= d_hat
            if d == group["d0"]:
                d = max(d, d_hat)

            # also don't let d "fall" below d, but limit with growth_rate
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

            # store everything back in the group
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = global_d_denom

        self._stats_buffer.push(
            group_idx = group_idx,
            name = group["name"],
            step = group["k"],
            d_num = group["d_numerator"],
            d_den = group["d_denom"],
            dlr = d * lr * bias_correction,
        )
        
        # buceta = group["name"]
        # if "dora" in buceta:
        #     print(f"tem alpha na {buceta}")

        # recompute dlr with the updated d (for this group)
        dlr = d * lr * bias_correction

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]
            exp_avg_sq = state["exp_avg_sq"]

            state["step"] += 1

            denom = exp_avg_sq.sqrt().add_(d * group["eps"])

            # Apply weight decay (decoupled variant)
            if group["weight_decay"] != 0 and decouple:
                p.data.add_(p.data, alpha=-group["weight_decay"] * dlr)

            ### Take step
            if beta1 > 0:
                exp_avg = state["exp_avg"]
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                    # <<< CHATGPT ADD STROCHASTIC PROTECT >>>
                    try:
                        addcdiv_stochastic_(p.data, exp_avg, denom, value=-dlr)
                    except Exception as e:
                        print(f"Stochastic rounding failed, using fallback addcdiv_: {e}")
                        traceback.print_exc()
                        p.data.addcdiv_(exp_avg, denom, value=-dlr)
                    # <<< FIM CHATGPT ADD STROCHASTIC PROTECT >>>
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
            else:
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                    # <<< CHATGPT ADD STROCHASTIC PROTECT <<<
                    try:
                        addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
                    except Exception as e:
                        print(f"Stochastic rounding failed, using fallback addcdiv_: {e}")
                        traceback.print_exc()
                        p.data.addcdiv_(grad, denom, value=-dlr * d)
                    # <<< FIM CHATGPT ADD STROCHASTIC PROTECT >>>
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)

        # Increment the group's k
        #print(f"[PRODIGY DEBUG] Group {group_idx} d={group['d']} d_numerator={group['d_numerator']} d_denom={group['d_denom']}")
        group["k"] = k + 1

    # --- INÍCIO DA LÓGICA DE ATUALIZAÇÃO COM WARMUP ---
    # Esta seção prepara os hiperparâmetros para a PRÓXIMA chamada.

    # Usaremos o contador do primeiro grupo como referência global para o warmup.
    # Isso assume que todos os grupos avançam juntos.
    # global_step = self.param_groups[0].get('k', 0) 
    # warmup_steps = getattr(self, 'dcoef_warmup_steps', 100) # Pega o valor do objeto optimizer

    # for group_idx, group in enumerate(self.param_groups):
    #     if not group['params'] or group['lr'] == 0.0:
    #         continue

    #     # 1. Medir a norma do gradiente (sempre)
    #     device = next((p.device for p in group['params'] if p.grad is not None), torch.device('cpu'))
    #     total_norm = torch.tensor(0.0, device=device)
    #     num_elements = 0
    #     for p in group['params']:
    #         if p.grad is not None:
    #             total_norm += p.grad.detach().abs().sum()
    #             num_elements += p.grad.numel()
        
    #     avg_grad_norm = (total_norm / (num_elements + 1e-9))
        
    #     # 2. Gerenciar o estado da EMA (sempre)
    #     dcoef_state = self.state.setdefault('_dcoef_state', {})
    #     ema_state_key = f'group_{group_idx}_ema_norm'
        
    #     # Se a EMA não existe, inicialize-a com a primeira medição.
    #     if ema_state_key not in dcoef_state:
    #         dcoef_state[ema_state_key] = avg_grad_norm
        
    #     ema_prev = dcoef_state[ema_state_key]
    #     beta_dc = group.get("dcoef_beta", 0.9)
    #     ema_curr = beta_dc * ema_prev + (1 - beta_dc) * avg_grad_norm
    #     dcoef_state[ema_state_key] = ema_curr # Sempre atualiza a EMA para mantê-la aquecida

    #     # 3. DECIDIR se vamos ATUALIZAR o d_coef
    #     if global_step > warmup_steps:
    #         # Warmup concluído. Agora podemos agir.
    #         target = group.get("dcoef_target", 0.05)
    #         dc_min = group.get("dcoef_min", 0.25)
    #         dc_max = group.get("dcoef_max", 4.0)

    #         new_dcoef = target / (ema_curr.item() + 1e-9)
    #         clamped_dcoef = torch.clamp(torch.tensor(new_dcoef), dc_min, dc_max).item()
            
    #         # ATUALIZA O VALOR NO GRUPO PARA A PRÓXIMA ITERAÇÃO
    #         group['d_coef'] = clamped_dcoef
            
    #         # Opcional: Logar quando a adaptação começa
    #         if global_step == warmup_steps + 1:
    #             print(f"INFO: DCoeff adaptation started for group {group_idx} at step {global_step}.")

    return loss


def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
    """
    Ativa o suporte a Stochastic Rounding no Prodigy,
    substituindo o método step pelo nosso step_prodigy.
    """
    try:
        optimizer.stochastic_rounding = stochastic_rounding
        optimizer.step = step_prodigy.__get__(optimizer, Prodigy)
        optimizer.pop_stats = _pop_prodigy_stats.__get__(optimizer, Prodigy)
    except Exception as e:
        print(f"Failed to set options in patch_prodigy: {e}")
        traceback.print_exc()

def _pop_prodigy_stats(self):
    """
    Retorna e esvazia o buffer das estatísticas mais recentes.
    Chame após optimizer.step().
    """
    return getattr(self, "_stats_buffer", []).pop_all()

class ProdigyStatsBuffer(list):
    """
    Coleciona dicts de estatísticas por grupo a cada .step().
    Chamou .pop_all() => devolve lista acumulada e limpa o buffer.
    """
    def push(self, **kwargs):
        self.append(kwargs)

    def pop_all(self):
        buf = list(self)
        self.clear()
        return buf