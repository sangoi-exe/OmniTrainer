import json
import gzip
import os

# Constantes para os nomes das métricas
METRIC_D_NUM = "d_num_pdgy"  # Como está no seu JSON grande
METRIC_D_DEN = "d_den_pdgy"  # Como está no seu JSON grande
OUTPUT_KEY_D_NUM = "d_num"  # Nome da chave no JSON de saída
OUTPUT_KEY_D_DEN = "d_den"  # Nome da chave no JSON de saída


def extract_num_den_for_all_modules_with_steps(  # Nome da função atualizado
        large_json_path: str,
        keys_list_filepath: str,
        output_num_den_json_path: str  # Nome do arquivo de saída atualizado
):
    """
    Parte 1: Lê um grande arquivo JSON de histórico de métricas e um arquivo de texto
    listando chaves de tensores. Extrai os nomes dos módulos base.
    Para cada módulo, extrai o histórico das métricas "d_numerator_global" e 
    "d_denominator_global" e os salva em um único arquivo JSON, onde cada par de valores 
    é associado ao seu step (índice). Retorna os dados processados.
    """
    if not os.path.exists(large_json_path):
        print(f"Erro: Arquivo JSON grande não encontrado em '{large_json_path}'")
        return None
    if not os.path.exists(keys_list_filepath):
        print(f"Erro: Arquivo de lista de chaves não encontrado em '{keys_list_filepath}'")
        return None

    output_directory = os.path.dirname(output_num_den_json_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    print(f"Lendo JSON grande de: '{large_json_path}'")
    print(f"Lendo lista de chaves (para identificar módulos) de: '{keys_list_filepath}'")
    print(f"Salvando dados de '{OUTPUT_KEY_D_NUM}' e '{OUTPUT_KEY_D_DEN}' com steps em: '{output_num_den_json_path}'")

    all_metrics_history = None
    try:
        open_func = gzip.open if large_json_path.endswith(".gz") else open
        mode = "rt" if large_json_path.endswith(".gz") else "r"
        with open_func(large_json_path, mode, encoding="utf-8") as f:
            print("  Carregando JSON grande na memória (pode demorar)...")
            data = json.load(f)
            all_metrics_history = data.get("metrics_history", {})
            print(f"  JSON grande carregado. {len(all_metrics_history)} entradas de módulo no arquivo de origem.")
    except MemoryError:
        print("ERRO DE MEMÓRIA: O arquivo JSON é muito grande para carregar na memória de uma vez.")
        return None
    except Exception as e:
        print(f"Erro ao carregar o arquivo JSON grande: {e}")
        return None

    if not all_metrics_history:
        print(f"Nenhum 'metrics_history' encontrado no JSON grande ou está vazio.")
        return None

    target_unique_base_module_names = set()
    with open(keys_list_filepath, "r", encoding="utf-8") as f_keys:
        for line in f_keys:
            line = line.strip()
            if line.startswith("===") or line == "(Empty)" or not line:
                continue

            layer_key_with_suffix = line
            potential_module_name = ""
            if layer_key_with_suffix.startswith("unet.tau_procs.") and layer_key_with_suffix.endswith(".log_tau"):
                potential_module_name = layer_key_with_suffix.replace("unet.tau_procs.", "").replace(".log_tau", "")
            else:
                parts = layer_key_with_suffix.split('.')
                if len(parts) > 1 and parts[-1] in ["alpha", "dora_scale", "weight"]:
                    if len(parts) > 2 and parts[-2] in ["lora_down", "lora_up"]:
                        potential_module_name = ".".join(parts[:-2])
                    else:
                        potential_module_name = ".".join(parts[:-1])
                else:
                    potential_module_name = layer_key_with_suffix

            if potential_module_name:
                target_unique_base_module_names.add(potential_module_name)

    print(f"Identificados {len(target_unique_base_module_names)} nomes de módulos base únicos de interesse.")

    num_den_data_with_steps = {}  # Dicionário para a nova estrutura
    num_modules_processed = 0

    for module_base_name in sorted(list(target_unique_base_module_names)):
        if module_base_name in all_metrics_history:
            module_specific_metrics = all_metrics_history[module_base_name]
            if METRIC_D_NUM in module_specific_metrics and METRIC_D_DEN in module_specific_metrics:
                d_num_values = module_specific_metrics[METRIC_D_NUM]
                d_den_values = module_specific_metrics[METRIC_D_DEN]

                # Garantir que ambas as listas tenham o mesmo comprimento
                # Se não tiverem, pode indicar um problema no registro de dados original.
                # Por agora, vamos usar o comprimento da menor lista para evitar IndexError.
                min_len = min(len(d_num_values), len(d_den_values))
                if len(d_num_values) != len(d_den_values):
                    print(
                        f"  Aviso: Para o módulo '{module_base_name}', '{METRIC_D_NUM}' ({len(d_num_values)}) e '{METRIC_D_DEN}' ({len(d_den_values)}) têm comprimentos diferentes. Usando {min_len}."
                    )

                num_den_data_with_steps[module_base_name] = [{
                    "step": i,
                    OUTPUT_KEY_D_NUM: d_num_values[i],
                    OUTPUT_KEY_D_DEN: d_den_values[i]} for i in range(min_len)]
                num_modules_processed += 1
            # else:
            # if METRIC_D_NUM not in module_specific_metrics:
            #     print(f"  Aviso: Métrica '{METRIC_D_NUM}' não encontrada para o módulo '{module_base_name}'.")
            # if METRIC_D_DEN not in module_specific_metrics:
            #     print(f"  Aviso: Métrica '{METRIC_D_DEN}' não encontrada para o módulo '{module_base_name}'.")

    if not num_den_data_with_steps:
        print(f"Nenhum dado para '{METRIC_D_NUM}' e '{METRIC_D_DEN}' encontrado para os módulos de interesse.")
        return None

    try:
        with open(output_num_den_json_path, "w", encoding="utf-8") as f_out:
            json.dump(num_den_data_with_steps, f_out, indent=4, ensure_ascii=False)
        print(f"\nProcessamento (Parte 1) concluído.")
        print(
            f"  Dados de '{OUTPUT_KEY_D_NUM}' e '{OUTPUT_KEY_D_DEN}' com steps para {num_modules_processed} módulos salvos em '{output_num_den_json_path}'."
        )
    except IOError as e:
        print(f"  Erro ao salvar o arquivo JSON (Parte 1): {e}")
        return None
    except Exception as e_gen:
        print(f"  Erro inesperado ao salvar (Parte 1): {e_gen}")
        return None

    return num_den_data_with_steps


def compress_num_den_sequences(  # Nome da função atualizado
        num_den_data_with_steps: dict, output_compressed_json_path: str):
    """
    Parte 2: Processa um dicionário de dados d_num/d_den (com steps individuais) e comprime
    sequências onde o PAR (d_num, d_den) é idêntico, marcando o intervalo de steps.
    Salva o resultado em um novo arquivo JSON.
    """
    if not num_den_data_with_steps:
        print("Nenhum dado para comprimir (Parte 2).")
        return

    output_directory_comp = os.path.dirname(output_compressed_json_path)
    if output_directory_comp:
        os.makedirs(output_directory_comp, exist_ok=True)

    print(f"\nIniciando Parte 2: Compressão de sequências de '{OUTPUT_KEY_D_NUM}/{OUTPUT_KEY_D_DEN}'.")
    print(f"Salvando JSON comprimido em: '{output_compressed_json_path}'")

    compressed_data = {}
    num_modules_compressed = 0

    for module_name, steps_data_list in num_den_data_with_steps.items():
        if not steps_data_list:
            compressed_data[module_name] = []
            continue

        compressed_module_data = []

        # Iniciar com o primeiro par de valores/step
        start_step = steps_data_list[0]["step"]
        current_d_num = steps_data_list[0][OUTPUT_KEY_D_NUM]
        current_d_den = steps_data_list[0][OUTPUT_KEY_D_DEN]

        for i in range(1, len(steps_data_list)):
            step_info = steps_data_list[i]
            # A sequência continua se AMBOS d_num e d_den forem iguais aos anteriores
            if step_info[OUTPUT_KEY_D_NUM] == current_d_num and \
               step_info[OUTPUT_KEY_D_DEN] == current_d_den:
                continue
            else:
                # Par de valores mudou, registrar o bloco anterior
                end_step = steps_data_list[i - 1]["step"]
                if start_step == end_step:
                    compressed_module_data.append({
                        "step": start_step,
                        OUTPUT_KEY_D_NUM: current_d_num,
                        OUTPUT_KEY_D_DEN: current_d_den})
                else:
                    compressed_module_data.append({
                        "steps": f"{start_step}-{end_step}",
                        OUTPUT_KEY_D_NUM: current_d_num,
                        OUTPUT_KEY_D_DEN: current_d_den})

                # Resetar para o novo par de valores/bloco
                start_step = step_info["step"]
                current_d_num = step_info[OUTPUT_KEY_D_NUM]
                current_d_den = step_info[OUTPUT_KEY_D_DEN]

        # Registrar o último bloco de valores
        end_step = steps_data_list[-1]["step"]
        if start_step == end_step:
            compressed_module_data.append({
                "step": start_step,
                OUTPUT_KEY_D_NUM: current_d_num,
                OUTPUT_KEY_D_DEN: current_d_den})
        else:
            compressed_module_data.append({
                "steps": f"{start_step}-{end_step}",
                OUTPUT_KEY_D_NUM: current_d_num,
                OUTPUT_KEY_D_DEN: current_d_den})

        compressed_data[module_name] = compressed_module_data
        num_modules_compressed += 1

    try:
        with open(output_compressed_json_path, "w", encoding="utf-8") as f_out:
            json.dump(compressed_data, f_out, indent=4, ensure_ascii=False)
        print(f"\nProcessamento (Parte 2) concluído.")
        print(
            f"  Dados de '{OUTPUT_KEY_D_NUM}/{OUTPUT_KEY_D_DEN}' comprimidos para {num_modules_compressed} módulos salvos em '{output_compressed_json_path}'."
        )
    except IOError as e:
        print(f"  Erro ao salvar o arquivo JSON comprimido (Parte 2): {e}")
    except Exception as e_gen:
        print(f"  Erro inesperado ao salvar (Parte 2): {e_gen}")


# --- Configuração Global ---
LARGE_JSON_FILE = r"F:\OT\data_recorder\Aracy.Cyb7.noDelta.003_Profile_Run1_20250513_135453.json"
KEYS_FILE = "unetKeysByBlock_20250512_215319.txt"
OUTPUT_NUM_DEN_WITH_STEPS_FILE = "todos_modulos_d_num_den_with_steps.json"
OUTPUT_NUM_DEN_COMPRESSED_FILE = "todos_modulos_d_num_den_compressed.json"

# --- Execução ---
if __name__ == "__main__":
    current_dir = os.getcwd()

    resolved_large_json_file = LARGE_JSON_FILE
    if not os.path.isabs(LARGE_JSON_FILE):
        resolved_large_json_file = os.path.join(current_dir, LARGE_JSON_FILE)
        if not os.path.exists(resolved_large_json_file):
            resolved_large_json_file = LARGE_JSON_FILE

    resolved_keys_file = KEYS_FILE
    if not os.path.isabs(KEYS_FILE):
        resolved_keys_file = os.path.join(current_dir, KEYS_FILE)
        if not os.path.exists(resolved_keys_file):
            resolved_keys_file = KEYS_FILE

    final_output_with_steps_path = os.path.join(current_dir, OUTPUT_NUM_DEN_WITH_STEPS_FILE)
    final_output_compressed_path = os.path.join(current_dir, OUTPUT_NUM_DEN_COMPRESSED_FILE)

    print(f"Usando arquivo JSON grande: {resolved_large_json_file}")
    print(f"Usando arquivo de chaves: {resolved_keys_file}")
    print(f"Arquivo de saída (steps individuais com d_num/d_den): {final_output_with_steps_path}")
    print(f"Arquivo de saída (comprimido com d_num/d_den): {final_output_compressed_path}")

    num_den_structured_data = extract_num_den_for_all_modules_with_steps(resolved_large_json_file, resolved_keys_file,
                                                                         final_output_with_steps_path)

    if num_den_structured_data:
        compress_num_den_sequences(num_den_structured_data, final_output_compressed_path)
    else:
        print("Parte 1 não retornou dados ou falhou, compressão (Parte 2) não será executada.")
