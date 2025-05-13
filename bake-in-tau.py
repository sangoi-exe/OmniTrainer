import argparse
import torch
from safetensors.torch import load_file, save_file

def bake_dora_safetensor_with_tau(
    safetensor_in: str,
    safetensor_out: str,
):
    """
    Carrega um safetensor que contém:
      - Pesos LoRA (ex.: lora_unet_..._to_q.lora_up.weight, .lora_down.weight, .alpha, etc.)
      - Parâmetros log_tau em keys como <prefix>_processor.log_tau
    Aplica tau=exp(log_tau) dividindo os pesos LoRA de `to_q` (e `to_k` se houver)
    e grava um novo safetensor com apenas os pesos ajustados.
    """
    print(f"🔄 Carregando safetensor: {safetensor_in}")
    sd = load_file(safetensor_in)

    new_sd = {}
    # Para cada par LoRA, aplica o tau correspondente
    for key, tensor in sd.items():
        # identificamos as chaves de LoRA up/down em to_q ou to_k
        if (".processor.to_q.lora_up.weight" in key or
            ".processor.to_q.lora_down.weight" in key or
            ".processor.to_k.lora_up.weight" in key or
            ".processor.to_k.lora_down.weight" in key):
            # extrai o prefixo do processor
            # ex: key="lora_unet_mid_block_..._attn1.processor.to_q.lora_up.weight"
            prefix = key.split("lora_unet_")[1]                      # "mid_block_..._attn1.processor.to_q.lora_up.weight"
            proc_key = prefix.split(".processor")[0] + "_processor.log_tau"
            tau_log = sd.get(proc_key)
            if tau_log is None:
                print(f"⚠️  Não achei τ para {key!r}, pulando sem alterar.")
                # new_sd[key] = tensor
            else:
                tau = torch.exp(torch.tensor(tau_log))              # converte log_tau → τ
                # expande τ para as dimensões do peso
                while tau.ndim < tensor.ndim:
                    tau = tau.unsqueeze(-1)
                # new_sd[key] = tensor / tau                          # baking: divide o peso por τ
                print(f"  ➕ Apliquei τ={tau.item():.4f} em {key}")
        else:
            # todo o resto copiamos direto (inclusive as entradas log_tau, alpha, etc.)
            new_sd[key] = tensor

    # (Opcional) remover as entradas de log_tau se não quiser incluí-las no arquivo final:
    keys_to_drop = [k for k in new_sd if k.endswith("processor.log_tau")]
    for k in keys_to_drop:
        del new_sd[k]

    print(f"💾 Salvando safetensor baked em: {safetensor_out}")
    save_file(new_sd, safetensor_out)
    print("✅ Bake-in completo!")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Bakes in τ nos pesos LoRA de um .safetensors único contendo tanto Dora quanto log_tau"
    )
    p.add_argument("--in",  dest="safetensor_in",  required=True, help="arquivo .safetensors de entrada")
    p.add_argument("--out", dest="safetensor_out", required=True, help="arquivo .safetensors de saída")
    args = p.parse_args()
    bake_dora_safetensor_with_tau(args.safetensor_in, args.safetensor_out)
