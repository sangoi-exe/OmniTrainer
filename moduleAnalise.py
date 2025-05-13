import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt # Para plotagem
import matplotlib.style as style # Para estilo de gráfico
from kneed import KneeLocator # pip install kneed
import re

# Aplicar um estilo mais agradável aos gráficos
style.use('seaborn-v0_8-whitegrid') # Ou 'ggplot', 'fivethirtyeight', etc.
plt.rcParams['figure.figsize'] = (15, 10) # Tamanho padrão da figura
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14


def plot_and_save_metrics(df: pd.DataFrame, module_name: str, summary: dict, output_plot_dir: str, config: dict):
    """
    Gera e salva um gráfico com as séries temporais das métricas e os elbows detectados.
    """
    if df.empty:
        print(f"  DataFrame vazio para {module_name}, pulando plotagem.")
        return

    safe_module_name_for_file = re.sub(r'[^\w.-]', '_', module_name)
    plot_filename = os.path.join(output_plot_dir, f"{safe_module_name_for_file}_metrics_plot.png")

    num_plots = 0
    if 'gd_ewma' in df.columns and not df['gd_ewma'].isnull().all(): num_plots +=1
    if 'gd_std_ewma_var' in df.columns and not df['gd_std_ewma_var'].isnull().all(): num_plots +=1
    if 'snr' in df.columns and not df['snr'].isnull().all(): num_plots +=1
    if 'gns_t' in df.columns and not df['gns_t'].isnull().all(): num_plots +=1
    
    if num_plots == 0:
        print(f"  Nenhuma métrica válida para plotar para {module_name}.")
        return

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), sharex=True)
    if num_plots == 1: # Se só um plot, axs não é uma lista
        axs = [axs]
    
    plot_idx = 0

    # Plot GD_EWMA e GD Bruto
    if 'gd_ewma' in df.columns and not df['gd_ewma'].isnull().all():
        ax = axs[plot_idx]
        gd_ewma_series = df['gd_ewma'].dropna()
        
        if 'gd' in df.columns and not df['gd'].isnull().all():
            ax.plot(df.index, df['gd'], label='GD (Bruto)', color='lightblue', alpha=0.6, linewidth=1)
        
        ax.plot(gd_ewma_series.index, gd_ewma_series, label='GD_EWMA', color='orange')
        
        if isinstance(summary.get('gd_ewma_elbow_initial_drop_step'), int):
            elbow_val = summary['gd_ewma_elbow_initial_drop_step']
            ax.vlines(elbow_val, gd_ewma_series.min(), gd_ewma_series.max(),
                       linestyles='--', colors='red', label=f'Elbow Inicial @{elbow_val}')
        if isinstance(summary.get('gd_ewma_plateau_step'), int):
            plateau_val = summary['gd_ewma_plateau_step']
            ax.vlines(plateau_val, gd_ewma_series.min(), gd_ewma_series.max(),
                       linestyles=':', colors='purple', label=f'Platô GD @{plateau_val}')
            
        ax.set_title(f"{module_name} - GD e GD_EWMA")
        ax.set_ylabel("Valor")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        plot_idx += 1

    # Plot sqrt(GD_std_ewma_var)
    if 'gd_std_ewma_var' in df.columns and not df['gd_std_ewma_var'].isnull().all():
        ax = axs[plot_idx]
        gd_std_dev_ewma = np.sqrt(df['gd_std_ewma_var'].astype(float).fillna(0))
        ax.plot(df.index, gd_std_dev_ewma, label='sqrt(GD_std_ewma_var)', color='green')
        
        if isinstance(summary.get('gd_std_ewma_stable_after_step'), int):
            stable_std_step = summary['gd_std_ewma_stable_after_step']
            ax.vlines(stable_std_step, gd_std_dev_ewma.min(), gd_std_dev_ewma.max(),
                       linestyles='--', colors='darkgreen', label=f'Std Estável @{stable_std_step}')

        ax.set_title(f"{module_name} - Desvio Padrão EWMA do GD")
        ax.set_ylabel("sqrt(Variância)")
        ax.set_yscale('log') # Frequentemente útil para std dev de GD
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        plot_idx += 1

    # Plot SNR
    if 'snr' in df.columns and not df['snr'].isnull().all():
        ax = axs[plot_idx]
        ax.plot(df.index, df['snr'], label='SNR', color='blue')
        ax.set_title(f"{module_name} - SNR (Tendência: {summary.get('snr_trend', 'N/A')})")
        ax.set_ylabel("SNR")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        plot_idx += 1

    # Plot GNS-T
    if 'gns_t' in df.columns and not df['gns_t'].isnull().all():
        ax = axs[plot_idx]
        ax.plot(df.index, df['gns_t'], label='GNS-T', color='red')
        ax.set_title(f"{module_name} - GNS-T (Tendência: {summary.get('gns_t_trend', 'N/A')})")
        ax.set_ylabel("GNS-T")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        # plot_idx += 1 # Não precisa mais incrementar aqui se for o último

    axs[-1].set_xlabel("Passo de Treino") # Adiciona label X apenas no último subplot
    plt.suptitle(f"Análise de Métricas para: {module_name}", fontsize=18, y=1.02) # Título geral
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Ajusta layout para não cortar o suptitle

    try:
        plt.savefig(plot_filename)
        # print(f"    -> Gráfico salvo em '{plot_filename}'")
    except Exception as e_plot:
        print(f"  ERRO ao salvar gráfico para {module_name}: {e_plot}")
    plt.close(fig) # Fecha a figura para liberar memória

def analyze_module_metrics(json_filepath: str, module_name: str, config: dict, output_plot_dir: str):
    """
    Analisa as métricas de um módulo, imprime um resumo e salva um gráfico.
    """
    print(f"\n--- Analisando Módulo: {module_name} ---")
    # ... (código de carregamento do JSON e criação do DataFrame como antes) ...
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  ERRO: Arquivo {json_filepath} não encontrado.")
        return
    except json.JSONDecodeError:
        print(f"  ERRO: Arquivo {json_filepath} não é um JSON válido.")
        return
    except Exception as e:
        print(f"  ERRO ao ler {json_filepath}: {e}")
        return

    try:
        # Tentar criar DataFrame, preenchendo com NaN se as listas tiverem tamanhos diferentes
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in data.items()]))
        # Adicionar um índice numérico se não houver um 'step' explícito
        if 'step' not in df.columns and not isinstance(df.index, pd.RangeIndex):
             df = df.reset_index().rename(columns={'index': 'step'})
        elif 'step' in df.columns:
            df = df.set_index('step', drop=False) # Usar 'step' como índice se existir

    except Exception as e_df:
        print(f"  ERRO ao criar DataFrame para {module_name}: {e_df}")
        expected_keys = ["gd", "gd_ewma", "gd_std_ewma_var", "snr", "gns_t", "d_pdgy"]
        valid_data = {}
        max_len = 0
        for k_test in expected_keys: # Encontrar o tamanho máximo para padding
            if k_test in data and isinstance(data[k_test], list):
                max_len = max(max_len, len(data[k_test]))

        for k in expected_keys:
            if k in data and isinstance(data[k], list):
                current_list = data[k]
                # Pad com NaN se a lista for menor que max_len
                padded_list = current_list + [np.nan] * (max_len - len(current_list))
                valid_data[k] = pd.Series(padded_list)
        if not valid_data:
            print("  Nenhuma das chaves esperadas foi encontrada como lista nos dados.")
            return
        df = pd.DataFrame(valid_data)
        if 'step' not in df.columns and not isinstance(df.index, pd.RangeIndex):
             df = df.reset_index().rename(columns={'index': 'step'})


    summary = {}
    # ... (toda a lógica de cálculo de summary para GD, SNR, GNS-T como antes) ...
    # 1. Análise do GD e GD_EWMA
    if 'gd_ewma' in df.columns and not df['gd_ewma'].isnull().all():
        gd_ewma_series = df['gd_ewma'].dropna()
        if len(gd_ewma_series) >= config.get("kneed_min_points", 10): # Reduzido para testes
            try:
                smoothed_gd_ewma = gd_ewma_series.rolling(window=max(1, config.get("kneed_smooth_window_gd", 5)), min_periods=1).mean()
                
                kl_decreasing = KneeLocator(
                    gd_ewma_series.index[smoothed_gd_ewma.index], # Usar índices originais para os pontos x
                    smoothed_gd_ewma.values, 
                    S=config.get("kneed_S_gd_decreasing", 1.0), 
                    curve='convex', 
                    direction='decreasing',
                    online=config.get("kneed_online_gd", False) # Online=False geralmente é melhor para achar "o" joelho
                )
                summary['gd_ewma_elbow_initial_drop_step'] = kl_decreasing.elbow if kl_decreasing.elbow is not None else "Não detectado"
                
                if kl_decreasing.elbow is not None:
                    stable_phase_start_idx_val = kl_decreasing.elbow + config.get("kneed_offset_after_elbow_gd", 20)
                    # Encontrar o índice real no DataFrame correspondente ao valor do step
                    possible_start_indices = gd_ewma_series.index[gd_ewma_series.index >= stable_phase_start_idx_val]
                    if not possible_start_indices.empty:
                        actual_stable_phase_start_index_loc = possible_start_indices[0]
                        gd_ewma_stable_phase = gd_ewma_series.loc[actual_stable_phase_start_index_loc:]
                        if not gd_ewma_stable_phase.empty:
                            summary['gd_ewma_stable_phase_mean'] = f"{gd_ewma_stable_phase.mean():.2e}"
                            summary['gd_ewma_stable_phase_median'] = f"{gd_ewma_stable_phase.median():.2e}"
                            summary['gd_ewma_stable_phase_std'] = f"{gd_ewma_stable_phase.std():.2e}"
                            
                            if len(gd_ewma_stable_phase) >= config.get("kneed_min_points", 10):
                                kl_plateau = KneeLocator(
                                    gd_ewma_stable_phase.index,
                                    gd_ewma_stable_phase.values,
                                    S=config.get("kneed_S_gd_plateau", 1.0), # Pode precisar de S diferente
                                    curve='convex', # Ainda decrescente, mas queremos onde a taxa de queda muda
                                    direction='decreasing',
                                    online=False
                                )
                                summary['gd_ewma_plateau_step'] = kl_plateau.elbow if kl_plateau.elbow is not None else "Não detectado"
                        else: summary['gd_ewma_stable_phase_info'] = "Fase estável vazia."
                    else: summary['gd_ewma_stable_phase_info'] = "Pico inicial ocupa demais."
            except Exception as e_kneed:
                summary['gd_ewma_elbow_error'] = f"Kneed GD: {str(e_kneed)}"
        else:
            summary['gd_ewma_elbow_info'] = "Poucos dados de gd_ewma."

    if 'gd_std_ewma_var' in df.columns and not df['gd_std_ewma_var'].isnull().all():
        gd_std_dev_ewma = np.sqrt(df['gd_std_ewma_var'].astype(float).fillna(0)) 
        low_std_threshold = config.get("gd_low_std_threshold", 1e-8) 
        # Encontra o primeiro índice onde a condição é verdadeira
        true_indices = df.index[gd_std_dev_ewma < low_std_threshold]
        stable_std_after_step = true_indices[0] if not true_indices.empty else "Nunca abaixo"
        summary['gd_std_ewma_stable_after_step'] = stable_std_after_step
        if isinstance(stable_std_after_step, (int, float)) and stable_std_after_step in gd_std_dev_ewma.index :
             summary['gd_std_ewma_median_stable_phase'] = f"{gd_std_dev_ewma.loc[stable_std_after_step:].median():.2e}"
        else:
            summary['gd_std_ewma_median_stable_phase'] = f"{gd_std_dev_ewma.median():.2e} (geral)"


    # 2. Análise do SNR
    if 'snr' in df.columns and not df['snr'].isnull().all():
        snr_series = df['snr'].dropna()
        if not snr_series.empty:
            summary['snr_mean'] = f"{snr_series.mean():.3f}"
            summary['snr_median'] = f"{snr_series.median():.3f}"
            summary['snr_std'] = f"{snr_series.std():.3f}"
            if len(snr_series) >= config.get("trend_min_points", 50):
                mid_idx = snr_series.index[len(snr_series) // 2]
                mean_first_half = snr_series.loc[:mid_idx].mean()
                mean_second_half = snr_series.loc[mid_idx:].mean()
                if mean_second_half < mean_first_half * (1 - config.get("trend_significant_change", 0.05)):
                    summary['snr_trend'] = "Decrescente"
                elif mean_second_half > mean_first_half * (1 + config.get("trend_significant_change", 0.05)):
                    summary['snr_trend'] = "Crescente"
                else:
                    summary['snr_trend'] = "Estável/Oscilante"
            else: summary['snr_trend'] = "Poucos dados"
        else: summary['snr_info'] = "Vazio"


    # 3. Análise do GNS-T
    if 'gns_t' in df.columns and not df['gns_t'].isnull().all():
        gns_t_series = df['gns_t'].dropna()
        if not gns_t_series.empty:
            summary['gns_t_mean'] = f"{gns_t_series.mean():.2f}"
            summary['gns_t_median'] = f"{gns_t_series.median():.2f}"
            summary['gns_t_std'] = f"{gns_t_series.std():.2f}"
            if len(gns_t_series) >= config.get("trend_min_points", 50):
                mid_idx = gns_t_series.index[len(gns_t_series) // 2]
                mean_first_half = gns_t_series.loc[:mid_idx].mean()
                mean_second_half = gns_t_series.loc[mid_idx:].mean()
                if mean_second_half > mean_first_half * (1 + config.get("trend_significant_change", 0.05)):
                    summary['gns_t_trend'] = "Crescente"
                elif mean_second_half < mean_first_half * (1 - config.get("trend_significant_change", 0.05)):
                    summary['gns_t_trend'] = "Decrescente"
                else:
                    summary['gns_t_trend'] = "Estável/Oscilante"
            else: summary['gns_t_trend'] = "Poucos dados"
        else: summary['gns_t_info'] = "Vazio"

    # Imprimir o resumo
    print("  Resumo das Métricas:")
    for key, value in summary.items():
        print(f"    {key}: {value}")

    # ... (lógica de sugestão de ponto de congelamento como antes) ...
    potential_freeze_points = []
    # Convertendo para int APENAS se for numérico e não string
    gd_elbow_initial = summary.get('gd_ewma_elbow_initial_drop_step')
    if isinstance(gd_elbow_initial, (int, float)) and not np.isnan(gd_elbow_initial) : potential_freeze_points.append(int(gd_elbow_initial))

    gd_plateau = summary.get('gd_ewma_plateau_step')
    if isinstance(gd_plateau, (int, float)) and not np.isnan(gd_plateau) : potential_freeze_points.append(int(gd_plateau))

    gd_std_stable = summary.get('gd_std_ewma_stable_after_step')
    if isinstance(gd_std_stable, (int, float)) and not np.isnan(gd_std_stable) : potential_freeze_points.append(int(gd_std_stable))
    
    if potential_freeze_points:
        freeze_candidate_gd = max(potential_freeze_points) if potential_freeze_points else "N/A"
        snr_ok_for_freeze = summary.get('snr_trend') in ["Decrescente", "Estável/Oscilante"]
        gns_t_ok_for_freeze = summary.get('gns_t_trend') in ["Crescente", "Estável/Oscilante"]

        if snr_ok_for_freeze and gns_t_ok_for_freeze and isinstance(freeze_candidate_gd, int):
            print(f"  Sugestão Heurística de Ponto de Congelamento: ~Step {freeze_candidate_gd}")
        elif isinstance(freeze_candidate_gd, int):
            print(f"  Ponto de estabilização do GD sugere ~Step {freeze_candidate_gd}, mas SNR/GNS-T podem não estar ideais.")
        else:
            print("  Não foi possível determinar um ponto de congelamento heurístico claro com GD.")
    else:
        print("  Não foi possível determinar um ponto de congelamento heurístico claro com GD.")

    # Gerar e salvar o gráfico
    if not df.empty:
        plot_and_save_metrics(df, module_name, summary, output_plot_dir, config)


# --- Configuração da Análise ---
analysis_config = {
    "kneed_min_points": 30,          # Reduzido para acomodar séries mais curtas após o primeiro elbow
    "kneed_smooth_window_gd": 10,    # Reduzido
    "kneed_S_gd_decreasing": 1.0,    
    "kneed_S_gd_plateau": 1.0,       # Ajustar sensibilidade para platô
    "kneed_online_gd": False,        
    "kneed_offset_after_elbow_gd": 50, # Quantos steps ignorar após o primeiro elbow
    "gd_low_std_threshold": 1e-8,    
    "trend_min_points": 50,         
    "trend_significant_change": 0.05 
}

# --- Execução ---
if __name__ == "__main__":
    module_jsons_dir = "lora_module_metric_history_jsons" 
    output_plots_dir = "lora_module_metric_plots" # Novo diretório para os gráficos
    os.makedirs(output_plots_dir, exist_ok=True)
    
    if not os.path.isdir(module_jsons_dir):
        print(f"ERRO: Diretório de entrada '{module_jsons_dir}' não encontrado.")
    else:
        all_files = [f for f in os.listdir(module_jsons_dir) if f.endswith("_metrics.json")]
        print(f"Encontrados {len(all_files)} arquivos de métricas de módulo para analisar.")
        
        limit_analysis_to_n_files = None # Defina um número para testar, ou None para todos
        
        for i, filename in enumerate(all_files):
            if limit_analysis_to_n_files is not None and i >= limit_analysis_to_n_files:
                print(f"\nAnálise limitada aos primeiros {limit_analysis_to_n_files} arquivos.")
                break
            
            module_name_from_filename = filename.replace("_metrics.json", "")
            filepath = os.path.join(module_jsons_dir, filename)
            analyze_module_metrics(filepath, module_name_from_filename, analysis_config, output_plots_dir)