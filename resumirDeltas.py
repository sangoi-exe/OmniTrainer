#!/usr/bin/env python3
# script_cli.py

# START: Substituição de interface gráfica por CLI – imports
import json
import os
import re
import sys
import argparse
# END: imports


# START: Função calculate_deltas sem GUI
def calculate_deltas(data, num_epochs_to_process, order):
    """
    Calcula a diferença (delta) dos valores dos módulos entre épocas consecutivas.
    """
    delta_results = {}

    # Extrair e ordenar as chaves de época numericamente
    epoch_keys = sorted(
        [key for key in data.keys() if re.match(r"epoch_\d+", key)],
        key=lambda x: int(re.search(r"\d+", x).group()))

    if not epoch_keys:
        print("Erro: nenhuma chave no formato 'epoch_X' encontrada no JSON de entrada.", file=sys.stderr)
        return None

    available_epoch_indices = [int(re.search(r"\d+", k).group()) for k in epoch_keys]
    total_epochs_available = len(epoch_keys)

    if num_epochs_to_process < 2:
        print("Erro: é necessário pelo menos 2 épocas para calcular deltas.", file=sys.stderr)
        return None
    if num_epochs_to_process > total_epochs_available:
        print(f"Erro: solicitadas {num_epochs_to_process} épocas, mas apenas {total_epochs_available} estão disponíveis.", file=sys.stderr)
        return None

    num_comparisons = num_epochs_to_process - 1

    if order == 'asc':
        # Compara (0->1), (1->2), ..., (N-2->N-1)
        for i in range(num_comparisons):
            idx_current = available_epoch_indices[i]
            idx_next = available_epoch_indices[i + 1]
            epoch_current_key = f"epoch_{idx_current}"
            epoch_next_key = f"epoch_{idx_next}"

            if epoch_current_key not in data or epoch_next_key not in data:
                continue

            data_current = data[epoch_current_key]
            data_next = data[epoch_next_key]
            delta_key = f"delta_{idx_current}_to_{idx_next}"
            delta_results[delta_key] = {}

            for module_name, value_current in data_current.items():
                if module_name in data_next:
                    delta_results[delta_key][module_name] = data_next[module_name] - value_current

    elif order == 'dsc':
        # Compara (M->M-1), (M-1->M-2), ...
        start_index = total_epochs_available - 1
        for i in range(num_comparisons):
            list_idx_current = start_index - i
            list_idx_prev = list_idx_current - 1
            if list_idx_prev < 0:
                break

            idx_current = available_epoch_indices[list_idx_current]
            idx_prev = available_epoch_indices[list_idx_prev]
            epoch_current_key = f"epoch_{idx_current}"
            epoch_prev_key = f"epoch_{idx_prev}"

            if epoch_current_key not in data or epoch_prev_key not in data:
                continue

            data_current = data[epoch_current_key]
            data_prev = data[epoch_prev_key]
            delta_key = f"delta_{idx_current}_to_{idx_prev}"
            delta_results[delta_key] = {}

            for module_name, value_current in data_current.items():
                if module_name in data_prev:
                    delta_results[delta_key][module_name] = data_prev[module_name] - value_current

    else:
        print(f"Erro: ordem de cálculo desconhecida: {order}", file=sys.stderr)
        return None

    if not delta_results:
        print("Aviso: nenhum delta calculado. Verifique o número de épocas e a ordem.", file=sys.stderr)
        return None

    return delta_results
# END: calculate_deltas


# START: Função de parsing de argumentos CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cálculo de deltas entre valores de módulos de JSON por época"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Caminho do arquivo JSON de entrada"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Caminho do arquivo JSON de saída"
    )
    parser.add_argument(
        "-n", "--num-epochs",
        type=int,
        required=True,
        help="Número de épocas a processar (>=2)"
    )
    parser.add_argument(
        "-r", "--order",
        choices=["asc", "dsc"],
        default="ascending",
        help="Ordem do cálculo de deltas"
    )
    return parser.parse_args()
# END: parse_args


# START: Bloco principal – fluxo via CLI
def main():
    args = parse_args()

    # Carregar JSON de entrada
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar JSON de entrada: {e}", file=sys.stderr)
        sys.exit(1)

    # Validar número de épocas
    if args.num_epochs < 2:
        print("Erro: --num-epochs deve ser pelo menos 2.", file=sys.stderr)
        sys.exit(1)

    # Determinar épocas disponíveis
    epoch_keys = [k for k in data.keys() if re.match(r"epoch_\d+", k)]
    total_available = len(epoch_keys)
    if args.num_epochs > total_available:
        print(f"Erro: solicitadas {args.num_epochs} épocas, mas apenas {total_available} disponíveis.", file=sys.stderr)
        sys.exit(1)

    # Calcular deltas
    result = calculate_deltas(data, args.num_epochs, args.order)
    if result is None:
        print("Falha no cálculo de deltas.", file=sys.stderr)
        sys.exit(1)

    # Salvar resultados
    try:
        out_dir = os.path.dirname(args.output) or '.'
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Sucesso: resultados salvos em {args.output}")
    except Exception as e:
        print(f"Erro ao salvar JSON de saída: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
# END: main
