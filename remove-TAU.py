import argparse
import torch
from safetensors.torch import load_file, save_file
import os

def remove_tau_keys_from_safetensor(
    safetensor_in: str,
    safetensor_out: str,
    tau_key_prefix: str = "unet.tau_procs.", # Prefixo comum para suas chaves tau
    tau_key_suffix: str = ".log_tau"        # Sufixo comum para suas chaves tau
):
    """
    Carrega um arquivo .safetensors, remove todas as chaves que correspondem
    ao padrão de chaves 'tau' e salva um novo arquivo .safetensors.

    Args:
        safetensor_in (str): Caminho para o arquivo .safetensors de entrada.
        safetensor_out (str): Caminho para o arquivo .safetensors de saída (sem chaves tau).
        tau_key_prefix (str): O prefixo que identifica as chaves de log_tau.
        tau_key_suffix (str): O sufixo que identifica as chaves de log_tau.
    """
    print(f"🔄 Carregando safetensor: {safetensor_in}")
    try:
        sd = load_file(safetensor_in)
        print(f"  Carregadas {len(sd)} chaves do arquivo de entrada.")
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo '{safetensor_in}': {e}")
        return

    keys_to_remove = []
    for key in sd.keys():
        is_tau_key = True
        # Verifica se a chave corresponde ao padrão de tau
        if tau_key_prefix and not key.startswith(tau_key_prefix):
            is_tau_key = False
        if tau_key_suffix and not key.endswith(tau_key_suffix):
            is_tau_key = False
        
        # Para ser mais específico, você pode querer que ambos, prefixo e sufixo, estejam presentes
        # se ambos forem fornecidos. Se apenas um for fornecido, ele será usado.
        if tau_key_prefix and tau_key_suffix: # Se ambos são definidos, ambos devem bater
            is_tau_key = key.startswith(tau_key_prefix) and key.endswith(tau_key_suffix)
        elif tau_key_prefix: # Só prefixo definido
             is_tau_key = key.startswith(tau_key_prefix)
        elif tau_key_suffix: # Só sufixo definido
             is_tau_key = key.endswith(tau_key_suffix)
        # Se nenhum for definido, nenhuma chave é considerada tau (o que não removeria nada)
        # mas o caso de uso aqui é remover chaves tau, então pelo menos um deve ser relevante.

        if is_tau_key:
            keys_to_remove.append(key)

    if not keys_to_remove:
        print("⚠️ Nenhuma chave correspondente ao padrão de tau foi encontrada para remover.")
        # Opcionalmente, você pode decidir copiar o arquivo ou não fazer nada.
        # Por segurança, vamos apenas informar e não criar um novo arquivo idêntico.
        # Se quiser copiar, descomente a linha abaixo e a de save_file.
        # print(f"   O arquivo de saída '{safetensor_out}' não será criado pois seria idêntico ao de entrada.")
        # return 
    else:
        print(f"  Encontradas {len(keys_to_remove)} chaves de tau para remover:")
        for k_rem in keys_to_remove[:10]: # Mostra as primeiras 10
            print(f"    - {k_rem}")
        if len(keys_to_remove) > 10:
            print(f"    ... e mais {len(keys_to_remove) - 10} chaves.")

    # Cria o novo state dict sem as chaves tau
    new_sd = {key: tensor for key, tensor in sd.items() if key not in keys_to_remove}

    print(f"💾 Salvando safetensor sem chaves tau em: {safetensor_out}")
    try:
        save_file(new_sd, safetensor_out)
        print(f"✅ Arquivo '{safetensor_out}' salvo com {len(new_sd)} chaves.")
    except Exception as e:
        print(f"❌ Erro ao salvar o arquivo '{safetensor_out}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove chaves de 'log_tau' de um arquivo .safetensors."
    )
    parser.add_argument("--in", dest="safetensor_in", required=True, 
                        help="Arquivo .safetensors de entrada (contendo chaves LoRA e log_tau).")
    parser.add_argument("--out", dest="safetensor_out", required=True, 
                        help="Arquivo .safetensors de saída (contendo apenas chaves LoRA).")
    parser.add_argument("--tau_prefix", dest="tau_prefix", default="unet.tau_procs.", 
                        help="Prefixo das chaves log_tau a serem removidas (ex: 'unet.tau_procs.').")
    parser.add_argument("--tau_suffix", dest="tau_suffix", default=".log_tau", 
                        help="Sufixo das chaves log_tau a serem removidas (ex: '.log_tau').")
    
    args = parser.parse_args()

    # Garante que pelo menos um critério de remoção (prefixo ou sufixo) foi fornecido
    # ou que o usuário intencionalmente deixou ambos vazios (o que não removeria nada).
    # No seu caso, os defaults são bons.
    
    remove_tau_keys_from_safetensor(
        args.safetensor_in, 
        args.safetensor_out,
        args.tau_prefix,
        args.tau_suffix
    )