import json
import gzip
import os
import re

def extract_metrics_per_module_base(
    large_json_path: str,
    keys_list_filepath: str, # O seu unetKeysByBlock_....txt
    output_module_jsons_dir: str
):
    """
    Lê um grande arquivo JSON de histórico de métricas e um arquivo de texto
    listando chaves de tensores. Extrai os nomes dos módulos base dessas chaves,
    encontra o histórico de métricas para cada módulo base único no JSON grande,
    e salva em um arquivo JSON individual por módulo base.

    Args:
        large_json_path (str): Caminho para o arquivo JSON.gz gigante.
        keys_list_filepath (str): Caminho para o arquivo .txt com a lista de chaves de parâmetros.
        output_module_jsons_dir (str): Diretório para salvar os JSONs por módulo base.
    """
    if not os.path.exists(large_json_path):
        print(f"Erro: Arquivo JSON grande não encontrado em '{large_json_path}'")
        return
    if not os.path.exists(keys_list_filepath):
        print(f"Erro: Arquivo de lista de chaves não encontrado em '{keys_list_filepath}'")
        return

    os.makedirs(output_module_jsons_dir, exist_ok=True)
    print(f"Lendo JSON grande de: '{large_json_path}'")
    print(f"Lendo lista de chaves de parâmetros de: '{keys_list_filepath}'")
    print(f"Salvando JSONs por módulo base em: '{output_module_jsons_dir}'")

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
        return
    except Exception as e:
        print(f"Erro ao carregar o arquivo JSON grande: {e}")
        return

    if not all_metrics_history:
        print("Nenhum 'metrics_history' encontrado no JSON grande ou está vazio.")
        return

    # Ler todas as chaves de parâmetros e extrair nomes de módulos base ÚNICOS
    unique_base_module_names = set()
    with open(keys_list_filepath, "r", encoding="utf-8") as f_keys:
        for line in f_keys:
            line = line.strip()
            if line.startswith("===") or line == "(Empty)" or not line:
                continue # Ignorar delimitadores e linhas vazias
            
            layer_key_with_suffix = line
            potential_module_name = ""

            if layer_key_with_suffix.startswith("unet.tau_procs.") and layer_key_with_suffix.endswith(".log_tau"):
                # Para os taus, o "módulo" é o nome do processador de atenção original (proc_key)
                potential_module_name = layer_key_with_suffix.replace("unet.tau_procs.", "").replace(".log_tau", "")
            else:
                # Para os LoRA/DoRA tradicionais
                parts = layer_key_with_suffix.split('.')
                # Tenta remover sufixos de parâmetros como .alpha, .dora_scale, .lora_down.weight, etc.
                if len(parts) > 1 and parts[-1] in ["alpha", "dora_scale", "weight"]:
                    if len(parts) > 2 and parts[-2] in ["lora_down", "lora_up"]:
                        potential_module_name = ".".join(parts[:-2])
                    else:
                        potential_module_name = ".".join(parts[:-1])
                else: 
                    # Se não for um sufixo conhecido, pode ser que a chave já seja o nome do módulo
                    # ou um tipo de parâmetro não esperado. Por segurança, consideramos a chave inteira.
                    # No entanto, para LoRA, é mais provável que seja um dos acima.
                    potential_module_name = layer_key_with_suffix 
            
            if potential_module_name:
                unique_base_module_names.add(potential_module_name)
    
    print(f"Encontrados {len(unique_base_module_names)} nomes de módulos base únicos para processar.")

    # Processar cada nome de módulo base único
    num_files_generated = 0
    num_modules_found_in_json = 0

    for module_base_name in sorted(list(unique_base_module_names)): # Ordenar para consistência
        if module_base_name in all_metrics_history:
            module_metrics_history = all_metrics_history[module_base_name]
            num_modules_found_in_json += 1

            safe_filename_base = re.sub(r'[^\w.-]', '_', module_base_name)
            output_filename = os.path.join(output_module_jsons_dir, f"{safe_filename_base}_metrics.json")

            try:
                with open(output_filename, "w", encoding="utf-8") as f_out:
                    json.dump(module_metrics_history, f_out, indent=4, ensure_ascii=False)
                num_files_generated += 1
                if num_files_generated % 50 == 0: # Log de progresso
                    print(f"  Processado e salvo {num_files_generated} arquivos de módulo...")

            except IOError as e:
                print(f"  Erro ao salvar JSON para o módulo '{module_base_name}': {e}")
            except Exception as e_gen:
                print(f"  Erro inesperado ao processar o módulo '{module_base_name}': {e_gen}")
        # else:
            # print(f"  Aviso: Módulo base '{module_base_name}' não encontrado no JSON grande.")

    print(f"\nProcessamento concluído.")
    print(f"  Total de nomes de módulos base únicos: {len(unique_base_module_names)}")
    print(f"  Total de módulos encontrados e processados do JSON grande: {num_modules_found_in_json}")
    print(f"  Total de arquivos JSON de módulo gerados: {num_files_generated} em '{output_module_jsons_dir}'.")

# --- Configuração ---
large_json_file = "caminho/para/seu/dump_de_155_milhoes.json.gz" # MUDE AQUI
keys_file = "unetKeysByBlock_20250512_215319.txt" 
output_per_module_dir = "lora_module_metric_history_jsons" # Novo nome de diretórioimport json
import gzip
import os
import re

def extract_metrics_per_module_base(
    large_json_path: str,
    keys_list_filepath: str, # O seu unetKeysByBlock_....txt
    output_module_jsons_dir: str
):
    """
    Lê um grande arquivo JSON de histórico de métricas e um arquivo de texto
    listando chaves de tensores. Extrai os nomes dos módulos base dessas chaves,
    encontra o histórico de métricas para cada módulo base único no JSON grande,
    e salva em um arquivo JSON individual por módulo base.

    Args:
        large_json_path (str): Caminho para o arquivo JSON.gz gigante.
        keys_list_filepath (str): Caminho para o arquivo .txt com a lista de chaves de parâmetros.
        output_module_jsons_dir (str): Diretório para salvar os JSONs por módulo base.
    """
    if not os.path.exists(large_json_path):
        print(f"Erro: Arquivo JSON grande não encontrado em '{large_json_path}'")
        return
    if not os.path.exists(keys_list_filepath):
        print(f"Erro: Arquivo de lista de chaves não encontrado em '{keys_list_filepath}'")
        return

    os.makedirs(output_module_jsons_dir, exist_ok=True)
    print(f"Lendo JSON grande de: '{large_json_path}'")
    print(f"Lendo lista de chaves de parâmetros de: '{keys_list_filepath}'")
    print(f"Salvando JSONs por módulo base em: '{output_module_jsons_dir}'")

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
        return
    except Exception as e:
        print(f"Erro ao carregar o arquivo JSON grande: {e}")
        return

    if not all_metrics_history:
        print("Nenhum 'metrics_history' encontrado no JSON grande ou está vazio.")
        return

    # Ler todas as chaves de parâmetros e extrair nomes de módulos base ÚNICOS
    unique_base_module_names = set()
    with open(keys_list_filepath, "r", encoding="utf-8") as f_keys:
        for line in f_keys:
            line = line.strip()
            if line.startswith("===") or line == "(Empty)" or not line:
                continue # Ignorar delimitadores e linhas vazias
            
            layer_key_with_suffix = line
            potential_module_name = ""

            if layer_key_with_suffix.startswith("unet.tau_procs.") and layer_key_with_suffix.endswith(".log_tau"):
                # Para os taus, o "módulo" é o nome do processador de atenção original (proc_key)
                potential_module_name = layer_key_with_suffix.replace("unet.tau_procs.", "").replace(".log_tau", "")
            else:
                # Para os LoRA/DoRA tradicionais
                parts = layer_key_with_suffix.split('.')
                # Tenta remover sufixos de parâmetros como .alpha, .dora_scale, .lora_down.weight, etc.
                if len(parts) > 1 and parts[-1] in ["alpha", "dora_scale", "weight"]:
                    if len(parts) > 2 and parts[-2] in ["lora_down", "lora_up"]:
                        potential_module_name = ".".join(parts[:-2])
                    else:
                        potential_module_name = ".".join(parts[:-1])
                else: 
                    # Se não for um sufixo conhecido, pode ser que a chave já seja o nome do módulo
                    # ou um tipo de parâmetro não esperado. Por segurança, consideramos a chave inteira.
                    # No entanto, para LoRA, é mais provável que seja um dos acima.
                    potential_module_name = layer_key_with_suffix 
            
            if potential_module_name:
                unique_base_module_names.add(potential_module_name)
    
    print(f"Encontrados {len(unique_base_module_names)} nomes de módulos base únicos para processar.")

    # Processar cada nome de módulo base único
    num_files_generated = 0
    num_modules_found_in_json = 0

    for module_base_name in sorted(list(unique_base_module_names)): # Ordenar para consistência
        if module_base_name in all_metrics_history:
            module_metrics_history = all_metrics_history[module_base_name]
            num_modules_found_in_json += 1

            safe_filename_base = re.sub(r'[^\w.-]', '_', module_base_name)
            output_filename = os.path.join(output_module_jsons_dir, f"{safe_filename_base}_metrics.json")

            try:
                with open(output_filename, "w", encoding="utf-8") as f_out:
                    json.dump(module_metrics_history, f_out, indent=4, ensure_ascii=False)
                num_files_generated += 1
                if num_files_generated % 50 == 0: # Log de progresso
                    print(f"  Processado e salvo {num_files_generated} arquivos de módulo...")

            except IOError as e:
                print(f"  Erro ao salvar JSON para o módulo '{module_base_name}': {e}")
            except Exception as e_gen:
                print(f"  Erro inesperado ao processar o módulo '{module_base_name}': {e_gen}")
        # else:
            # print(f"  Aviso: Módulo base '{module_base_name}' não encontrado no JSON grande.")

    print(f"\nProcessamento concluído.")
    print(f"  Total de nomes de módulos base únicos: {len(unique_base_module_names)}")
    print(f"  Total de módulos encontrados e processados do JSON grande: {num_modules_found_in_json}")
    print(f"  Total de arquivos JSON de módulo gerados: {num_files_generated} em '{output_module_jsons_dir}'.")

# --- Configuração ---
output_per_module_dir = "EUVOMATAOVSCODE" # Novo nome de diretório
# --- Configuração ---
# O JSON gigante do DataRecorder
large_json_file = r"F:\OT\data_recorder\Sil.Cyb7.noDelta.001_Profile_Run1_20250512_212356.json" # MUDE AQUI
# O arquivo TXT com a lista de todas as chaves (layers) que você quer extrair
keys_file = "unetKeysByBlock_20250512_215319.txt" 
# Diretório onde os JSONs individuais por layer serão salvos
# --- Execução ---
if __name__ == "__main__":
    if not os.path.isabs(large_json_file):
        large_json_file = os.path.join(os.getcwd(), large_json_file)
    if not os.path.isabs(keys_file):
        keys_file = os.path.join(os.getcwd(), keys_file)
    if not os.path.isabs(output_per_module_dir):
        output_per_module_dir = os.path.join(os.getcwd(), output_per_module_dir)
        
    extract_metrics_per_module_base(large_json_file, keys_file, output_per_module_dir)

# --- Execução ---
if __name__ == "__main__":
    if not os.path.isabs(large_json_file):
        large_json_file = os.path.join(os.getcwd(), large_json_file)
    if not os.path.isabs(keys_file):
        keys_file = os.path.join(os.getcwd(), keys_file)
    if not os.path.isabs(output_per_module_dir):
        output_per_module_dir = os.path.join(os.getcwd(), output_per_module_dir)
        
    extract_metrics_per_module_base(large_json_file, keys_file, output_per_module_dir)