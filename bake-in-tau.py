import argparse
import torch
from safetensors.torch import load_file, save_file
import math
import os
from datetime import datetime
# import re # Removido, pois a lógica de string substitui a regex complexa por enquanto

def bake_lora_with_tau_effect_final_attempt(
    safetensor_in: str,
    safetensor_out: str,
    log_tau_amplifier: float = 1000.0,
    log_filename_prefix: str = "bake_tau_log_final"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Adiciona o fator de amplificação ao nome do arquivo de log para clareza
    debug_log_filename = f"{log_filename_prefix}_amp{log_tau_amplifier:.1f}_{timestamp}.txt" 
    debug_log_lines = []

    def add_to_debug_log(message):
        print(message) # Continua printando no console para feedback imediato
        debug_log_lines.append(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}")

    add_to_debug_log(f"Iniciando bake_lora_with_tau_effect para: {safetensor_in}")
    add_to_debug_log(f"Fator de amplificação para log_tau: {log_tau_amplifier}")
    add_to_debug_log(f"Arquivo de saída planejado: {safetensor_out}")
    add_to_debug_log(f"Log de debug detalhado será salvo em: {os.path.abspath(debug_log_filename)}")

    try:
        sd = load_file(safetensor_in)
        add_to_debug_log(f"  Carregadas {len(sd)} chaves do arquivo de entrada.")
    except Exception as e:
        add_to_debug_log(f"ERRO FATAL ao carregar {safetensor_in}: {e}")
        with open(debug_log_filename, "w", encoding="utf-8") as f_log:
            f_log.write("\n".join(debug_log_lines))
        return

    new_sd = {k: v.clone() for k, v in sd.items()}
    # Chave: proc_key_original_tau_format (com pontos e _processor, 
    # ex: "down_blocks.1.attentions.0.transformer_blocks.0.attn1_processor")
    attention_layer_scale_factors = {} 
    log_tau_keys_to_remove = []

    # --- PASSO 1: Coletar log_taus, aplicar amplificador, e calcular fatores de escala ---
    add_to_debug_log("\n--- PASSO 1: Coletando log_taus, aplicando amplificador e calculando fatores de escala ---")
    for key, tensor_val in sd.items():
        if key.startswith("unet.tau_procs.") and key.endswith(".log_tau"):
            # proc_key_original_tau_format é o identificador da camada de atenção como usado nas chaves tau
            # Ex: "down_blocks.1.attentions.0.transformer_blocks.0.attn1_processor"
            proc_key_original_tau_format = key.replace("unet.tau_procs.", "").replace(".log_tau", "")
            
            original_log_tau_tensor = tensor_val
            # Aplica o fator de amplificação aqui
            amplified_log_tau_tensor = original_log_tau_tensor * log_tau_amplifier 
            
            tau_val = torch.exp(amplified_log_tau_tensor) # Usa o log_tau amplificado
            scale_factor = 1.0 / torch.sqrt(tau_val)
            
            attention_layer_scale_factors[proc_key_original_tau_format] = scale_factor
            log_tau_keys_to_remove.append(key) # Guarda a chave original do log_tau para remoção
            add_to_debug_log(f"  [TAU_COLLECT] TauKey Formato Original: '{proc_key_original_tau_format}'")
            add_to_debug_log(f"    Original log_tau: {original_log_tau_tensor.item():.4f}")
            add_to_debug_log(f"    Amplified log_tau: {amplified_log_tau_tensor.item():.4f} (x{log_tau_amplifier})")
            add_to_debug_log(f"    Resulting tau: {tau_val.item():.4f}, scale_factor (1/sqrt(tau)): {scale_factor.item():.4f}")

    if not attention_layer_scale_factors:
        add_to_debug_log("⚠️ Nenhum parâmetro log_tau (com prefixo 'unet.tau_procs.') foi encontrado.")
        # Remove quaisquer outras chaves tau genéricas se existirem e salva
        keys_to_remove_anyway = [k for k_sd in new_sd for k in [k_sd] if ".log_tau" in k or ".log_tau_param" in k] # Garante que k é string
        if keys_to_remove_anyway:
            add_to_debug_log(f"  Removendo {len(keys_to_remove_anyway)} chaves tau genéricas encontradas...")
            for k_rem in keys_to_remove_anyway: 
                if k_rem in new_sd: 
                    del new_sd[k_rem]
                    add_to_debug_log(f"    Removida chave genérica: {k_rem}")
        save_file(new_sd, safetensor_out)
        add_to_debug_log(f"✅ Processamento (sem bake-in de tau ativo) completo! Saída: {safetensor_out}")
        with open(debug_log_filename, "w", encoding="utf-8") as f_log: f_log.write("\n".join(debug_log_lines))
        return
    else:
        add_to_debug_log(f"  Total de {len(attention_layer_scale_factors)} fatores de escala de tau calculados.")
        # add_to_debug_log(f"DEBUG: Chaves de tau coletadas (formato original para dict): {list(attention_layer_scale_factors.keys())[:5]}")

    # --- PASSO 2: Aplicar os scale_factors aos pesos LoRA de Q e K ---
    add_to_debug_log("\n--- PASSO 2: Aplicando fatores de escala aos pesos LoRA de Atenção (Q/K) ---")
    lora_weights_modified_count = 0
    unmatched_lora_q_k_keys_info = [] # Para armazenar info sobre chaves não correspondidas

    for lora_key in list(new_sd.keys()): # Iterar sobre cópia das chaves para poder modificar new_sd
        if not lora_key.startswith("lora_unet_"):
            continue

        key_no_lora_prefix = lora_key[len("lora_unet_"):]
        
        attention_block_slug_from_lora = None # Formato esperado: "down_blocks_X_attentions_Y_transformer_blocks_Z_attnN" (underscores)
        is_q_or_k_lora_weight = False
        lora_component_type = "" # "lora_up.weight" ou "lora_down.weight"

        # Tenta extrair o slug e identificar se é Q ou K e qual componente LoRA (up/down)
        if key_no_lora_prefix.endswith("_to_q.lora_up.weight"):
            attention_block_slug_from_lora = key_no_lora_prefix[:-len("_to_q.lora_up.weight")]
            is_q_or_k_lora_weight = True
            lora_component_type = "to_q.lora_up.weight"
        elif key_no_lora_prefix.endswith("_to_q.lora_down.weight"):
            attention_block_slug_from_lora = key_no_lora_prefix[:-len("_to_q.lora_down.weight")]
            is_q_or_k_lora_weight = True
            lora_component_type = "to_q.lora_down.weight"
        elif key_no_lora_prefix.endswith("_to_k.lora_up.weight"):
            attention_block_slug_from_lora = key_no_lora_prefix[:-len("_to_k.lora_up.weight")]
            is_q_or_k_lora_weight = True
            lora_component_type = "to_k.lora_up.weight"
        elif key_no_lora_prefix.endswith("_to_k.lora_down.weight"):
            attention_block_slug_from_lora = key_no_lora_prefix[:-len("_to_k.lora_down.weight")]
            is_q_or_k_lora_weight = True
            lora_component_type = "to_k.lora_down.weight"
        
        if is_q_or_k_lora_weight and attention_block_slug_from_lora:
            # attention_block_slug_from_lora é como: "down_blocks_2_attentions_0_transformer_blocks_0_attn1"
            
            # Converter o slug (com underscores) para o formato da chave tau (com pontos)
            # E adicionar o sufixo "_processor"
            # Ex: "down_blocks_2_attentions_0_transformer_blocks_0_attn1" 
            #  -> "down_blocks.2.attentions.0.transformer_blocks.0.attn1_processor"
            proc_key_lookup_for_tau = attention_block_slug_from_lora + "_processor"
            
            # Log para cada tentativa de match
            # add_to_debug_log(f"  [LORA_PROC] LoRA Key: '{lora_key}' -> Slug LoRA: '{attention_block_slug_from_lora}' -> Chave Tau Lookup: '{proc_key_lookup_for_tau}'")
            
            if proc_key_lookup_for_tau in attention_layer_scale_factors:
                scale_factor_to_apply = attention_layer_scale_factors[proc_key_lookup_for_tau]
                lora_tensor = new_sd[lora_key] # Pega o tensor para modificar da cópia
                original_dtype = lora_tensor.dtype
                
                # Modifica o tensor diretamente no dicionário new_sd
                new_sd[lora_key] = (lora_tensor.float() * scale_factor_to_apply.float()).to(original_dtype)
                
                add_to_debug_log(f"    ✅ BAKED τ em '{lora_key}' (usando TauKey '{proc_key_lookup_for_tau}') com scale_factor {scale_factor_to_apply.item():.4f}")
                lora_weights_modified_count += 1
            else:
                # Adiciona à lista de não correspondidos apenas se for um Q/K de atenção
                if "attentions" in attention_block_slug_from_lora: # Filtro adicional para ter certeza
                     unmatched_lora_q_k_keys_info.append(f"LoRA Key: '{lora_key}' -> Slug: '{attention_block_slug_from_lora}' -> Tau Lookup: '{proc_key_lookup_for_tau}' (NÃO ENCONTRADO)")
        # else:
            # add_to_debug_log(f"  Skipping non-Q/K LoRA weight key or pattern mismatch: {lora_key}")

    if lora_weights_modified_count > 0:
        add_to_debug_log(f"\n  Total de {lora_weights_modified_count} pesos LoRA (para Q/K) tiveram tau baked-in.")
    elif attention_layer_scale_factors: # Se tínhamos taus mas não modificamos nenhum peso LoRA
        add_to_debug_log("\n  AVISO FINAL: Fatores de escala de tau foram calculados, mas nenhum peso LoRA de atenção Q/K foi modificado.")
    
    if unmatched_lora_q_k_keys_info:
        add_to_debug_log(f"\n  AVISO: {len(unmatched_lora_q_k_keys_info)} chaves LoRA de atenção Q/K não encontraram um tau mapeado correspondente:")
        for unmatched_info in unmatched_lora_q_k_keys_info[:20]: # Mostra as primeiras 20
            add_to_debug_log(f"    - {unmatched_info}")
        if len(unmatched_lora_q_k_keys_info) > 20:
            add_to_debug_log(f"    ... e mais {len(unmatched_lora_q_k_keys_info) - 20} chaves não correspondidas.")
        add_to_debug_log(f"    Exemplo de chave de tau REAL que foi coletada (para comparação de formato): '{list(attention_layer_scale_factors.keys())[0]}'")


    # --- PASSO 3: Remover as chaves log_tau originais ---
    if log_tau_keys_to_remove:
        add_to_debug_log(f"\n--- PASSO 3: Removendo chaves log_tau ---")
        add_to_debug_log(f"  Removendo {len(log_tau_keys_to_remove)} chaves de log_tau do safetensor final...")
        for k_to_drop in log_tau_keys_to_remove:
            if k_to_drop in new_sd: # Verifica se a chave ainda existe antes de deletar
                del new_sd[k_to_drop]
                add_to_debug_log(f"    Removida chave: {k_to_drop}")
    
    add_to_debug_log(f"\n💾 Salvando safetensor baked em: {safetensor_out}")
    try:
        save_file(new_sd, safetensor_out)
        add_to_debug_log("✅ Bake-in completo!")
    except Exception as e_save:
        add_to_debug_log(f"ERRO FATAL ao salvar {safetensor_out}: {e_save}")
    finally:
        # Garante que o log de debug seja escrito mesmo se houver erro no save_file
        try:
            with open(debug_log_filename, "w", encoding="utf-8") as f_log:
                f_log.write("\n".join(debug_log_lines))
            print(f"📝 Log de debug detalhado salvo em: {os.path.abspath(debug_log_filename)}")
        except Exception as e_log_write:
            print(f"ERRO CRÍTICO ao escrever arquivo de log de debug '{debug_log_filename}': {e_log_write}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aplica (bakes in) o efeito de τ (de log_tau) aos pesos LoRA de Q e K de um .safetensors, com amplificação opcional."
    )
    parser.add_argument("--in",  dest="safetensor_in",  required=True, help="Arquivo .safetensors de entrada.")
    parser.add_argument("--out", dest="safetensor_out", required=True, help="Arquivo .safetensors de saída.")
    parser.add_argument("--amplify_log_tau", type=float, default=1.0, dest="amplify_factor",
                        help="Fator para multiplicar os valores de log_tau antes de calcular tau (ex: 10.0 para exagerar o efeito). Padrão: 1.0 (sem amplificação).")
    parser.add_argument("--logprefix", dest="log_prefix", default="bake_tau_log", help="Prefixo para o nome do arquivo de log de debug.")
    
    args = parser.parse_args()
    
    bake_lora_with_tau_effect_final_attempt(
        args.safetensor_in, 
        args.safetensor_out,
        args.amplify_factor,
        args.log_prefix
    )