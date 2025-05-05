import json
import re
import os

def sanitize_filename(filename):
    """Remove ou substitui caracteres inválidos para nomes de arquivo."""
    # Remove caracteres inválidos no Windows e Linux/macOS
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Limita o comprimento para evitar problemas em alguns sistemas de arquivos
    max_len = 200
    if len(sanitized) > max_len:
        # Tenta manter o final do nome, que pode ser mais descritivo
        sanitized = sanitized[-max_len:]
    return sanitized

# --- Configuração ---
# Use os.path.join para compatibilidade entre sistemas operacionais
# Substitua 'seu_diretorio' pelo caminho onde o script e o JSON estão, se necessário.
# Se estiverem no mesmo diretório, pode deixar ''
diretorio_atual = ''
input_filename = os.path.join(diretorio_atual, 'Aracy.Cyb7.noDelta.001_Deltas_Run1_20250502_210608.json')
# --- Fim Configuração ---

# 1. Obter o nome exato do módulo do usuário
target_module = input("Digite o nome exato do módulo que deseja extrair: ")

# 2. Ler o arquivo JSON de entrada
try:
    with open(input_filename, 'r', encoding='utf-8') as infile:
        all_data = json.load(infile)
    print(f"Arquivo '{input_filename}' lido com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo de entrada '{input_filename}' não encontrado.")
    exit(1)
except json.JSONDecodeError:
    print(f"Erro: O arquivo '{input_filename}' não contém JSON válido.")
    exit(1)
except Exception as e:
    print(f"Ocorreu um erro inesperado ao ler o arquivo: {e}")
    exit(1)

# 3. Preparar a estrutura de dados de saída
output_data = {}
found_module_in_any_epoch = False

# 4. Iterar pelas épocas e filtrar os dados para o módulo alvo
print(f"Extraindo dados para o módulo: '{target_module}'...")
for epoch_key, epoch_data in all_data.items():
    # Garante que os dados da época sejam um dicionário antes de prosseguir
    if isinstance(epoch_data, dict):
        # Verifica se o módulo alvo existe nesta época
        if target_module in epoch_data:
            # Se existir, cria a entrada da época no output_data (se ainda não existir)
            # e adiciona o módulo e seu valor
            if epoch_key not in output_data:
                 output_data[epoch_key] = {}
            output_data[epoch_key][target_module] = epoch_data[target_module]
            found_module_in_any_epoch = True
        # else:
            # Opcional: Se quiser que as épocas sem o módulo apareçam vazias no JSON final
            # if epoch_key not in output_data:
            #     output_data[epoch_key] = {}
    else:
        print(f"Aviso: Dados para a época '{epoch_key}' não são um dicionário. Ignorando.")


# 5. Verificar se algum dado foi encontrado para o módulo
if not found_module_in_any_epoch:
    print(f"Aviso: Nenhum dado encontrado para o módulo '{target_module}' em nenhuma época.")
    # Você pode decidir sair aqui ou criar um arquivo JSON vazio
    # exit() # Descomente para sair se nada for encontrado

# 6. Gerar um nome de arquivo de saída seguro
safe_module_name = sanitize_filename(target_module)
output_filename = os.path.join(diretorio_atual, f"{safe_module_name}_epochs.json")

# 7. Escrever os dados filtrados no novo arquivo JSON
try:
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # Usa indent=4 para formatação legível e ensure_ascii=False para caracteres não-ASCII
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)
    print(f"\nDados extraídos com sucesso e salvos em '{output_filename}'")
except Exception as e:
    print(f"Ocorreu um erro ao escrever o arquivo de saída: {e}")
    exit(1)