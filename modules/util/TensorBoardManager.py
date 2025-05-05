import time
import torch
from threading import Lock
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Optional, Tuple, List, Union

from modules.sangoi.logFun import logFun


class TensorBoardManager:
    """
    Um wrapper para torch.utils.tensorboard.SummaryWriter que acumula
    dados na memória e os escreve no disco de forma agrupada para
    reduzir a contenção de I/O.
    """

    def __init__(
        self, log_dir: str, flush_secs: int = 30, max_queue_size: int = 10000
    ):
        """
        Inicializa o TensorBoardManager.

        Args:
            log_dir (str): Diretório para salvar os logs do TensorBoard.
            flush_secs (int): Tempo máximo em segundos entre escritas no disco.
                              (Não usado na implementação por época, mas útil para alternativas)
            max_queue_size (int): Tamanho máximo da fila de eventos antes de forçar
                                  uma escrita (para evitar consumo excessivo de RAM).
        """
        # Cria o diretório se não existir (SummaryWriter pode fazer isso, mas garantimos)
        import os

        os.makedirs(log_dir, exist_ok=True)

        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs * 2)
        
        self._log_dir = log_dir
        self._flush_secs = flush_secs
        self._max_queue_size = max_queue_size

        # Buffers para diferentes tipos de dados
        self._scalar_buffer: List[Tuple[str, Any, int]] = []
        self._image_buffer: List[Tuple[str, Any, int, str]] = []
        
        self._scalar_buffer: List[Tuple[str, Union[float, int], int]] = []
        self._image_buffer: List[Tuple[str, np.ndarray, int, str]] = []

        self._buffer_lock = Lock()
        self._last_flush_time = time.time()

        self._total_items_in_buffer = 0

        logFun(f"[TensorBoardManager] Inicializado para o diretório: {log_dir}", lvl="success")
        logFun(f"[TensorBoardManager] Flush automático a cada {flush_secs}s ou {max_queue_size} itens.", lvl="info")

    def add_scalar(
        self, tag: str, scalar_value: Any, global_step: Optional[int] = None
    ):
        
        """Adiciona um escalar ao buffer."""
        # Garante que o valor seja um tipo numérico básico (float/int)
        value: Union[float, int]
        if isinstance(scalar_value, torch.Tensor):
            value = scalar_value.detach().cpu().item()
        elif isinstance(scalar_value, (int, float)):
            value = scalar_value
        else:
              try:
                  value = float(scalar_value)
              except (ValueError, TypeError):
                  warnings.warn(f"[TensorBoardManager] Valor escalar para '{tag}' não é numérico ({type(scalar_value)}). Ignorando.")
                  return

        with self._buffer_lock:
            # START: Modificações em add_scalar
            self._scalar_buffer.append((tag, value, global_step))
            self._total_items_in_buffer += 1
            self._check_flush_conditions() # Verifica se precisa fazer flush
            # END: Modificações em add_scalar

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        dataformats="CHW",
    ):
        """Adiciona uma imagem ao buffer."""
        if not isinstance(global_step, int):
            global_step = 0 # Default step
        # START: Processamento da imagem antes de adicionar ao buffer
        processed_image: Optional[np.ndarray] = None
        try:
            if isinstance(img_tensor, torch.Tensor):
                # Move para CPU, desanexa, converte para NumPy
                processed_image = img_tensor.detach().cpu().numpy()
            elif isinstance(img_tensor, np.ndarray):
                # Assume que já está no formato correto (CPU)
                processed_image = img_tensor
            else:
                 warnings.warn(f"[TensorBoardManager] Tipo de imagem não suportado para '{tag}': {type(img_tensor)}. Use torch.Tensor ou np.ndarray. Ignorando.")
                 return

            # Validação básica do formato de dados (opcional, mas útil)
            if dataformats not in ['CHW', 'HWC', 'HW']:
                 warnings.warn(f"[TensorBoardManager] Formato de dados de imagem inválido '{dataformats}' para tag '{tag}'. Use 'CHW', 'HWC' ou 'HW'. Ignorando.")
                 return

        except Exception as e:
            warnings.warn(f"[TensorBoardManager] Erro ao processar imagem para '{tag}': {e}. Ignorando.")
            return # Não adiciona ao buffer se o processamento falhar

        if processed_image is None:
             return # Sai se não foi possível processar
        # Armazena o tensor diretamente para evitar cópias desnecessárias agora.
        # A conversão/processamento ocorrerá durante o flush.
        # Nota: Isso pode consumir RAM se muitas imagens grandes forem logadas.
        with self._buffer_lock:
            self._image_buffer.append((tag, processed_image, global_step, dataformats))
            self._total_items_in_buffer += 1
            self._check_flush_conditions() # Verifica se precisa fazer flush
        # END: Processamento da imagem antes de adicionar ao buffer

    # START: Lógica de verificação de flush (tamanho e tempo)
    def _check_flush_conditions(self, force: bool = False):
        """
        Verifica se o buffer precisa ser escrito baseado no tamanho ou tempo.
        Esta função DEVE ser chamada com o _buffer_lock adquirido.
        """
        now = time.time()
        should_flush = False

        # 1. Condição de Flush Forçado (externo)
        if force:
             should_flush = True
             # print("[TensorBoardManager] Flush forçado.") # Debug

        # 2. Condição de Tamanho Máximo da Fila
        elif self._total_items_in_buffer >= self._max_queue_size:
             should_flush = True
             # print(f"[TensorBoardManager] Flush devido ao tamanho da fila ({self._total_items_in_buffer}/{self._max_queue_size}).") # Debug

        # 3. Condição de Tempo desde o Último Flush
        elif (now - self._last_flush_time) >= self._flush_secs:
             should_flush = True
             # print(f"[TensorBoardManager] Flush devido ao tempo ({now - self._last_flush_time:.1f}s >= {self._flush_secs}s).") # Debug

        # Executa o flush se qualquer condição for atendida E houver dados no buffer
        if should_flush and self._total_items_in_buffer > 0:
            self._flush_buffers() # _flush_buffers já lida com a escrita e reset
    # END: Lógica de verificação de flush (tamanho e tempo)

    def _check_flush_needed(self, force: bool = False):
        """Verifica se o buffer precisa ser escrito (baseado no tamanho)."""
        # Nota: A lógica de flush por tempo foi removida em favor do flush por época.
        # Esta verificação agora serve apenas para evitar estouro de RAM.
        if force:
            # print(f"[TensorBoardManager] Flush forçado devido ao tamanho do buffer.") # Debug
            # Idealmente, não deveríamos fazer flush aqui, mas logar um aviso
            # ou implementar uma estratégia mais robusta (ex: descartar dados antigos).
            # Por ora, vamos apenas avisar se o buffer estiver muito grande.
            if (
                len(self._scalar_buffer) > self._max_queue_size
                or len(self._image_buffer) > self._max_queue_size
            ):
                print(
                    f"[TensorBoardManager] Atenção: Buffer próximo do limite máximo ({self._max_queue_size}). Considere aumentar o limite ou a frequência de flush."
                )
                # Poderia forçar flush aqui se necessário: self._flush_buffers()
                pass

    def _flush_buffers(self):
        """
        Escreve o conteúdo dos buffers no disco usando o SummaryWriter.
        Esta função DEVE ser chamada com o _buffer_lock adquirido.
        """
        if self._total_items_in_buffer == 0: # Verifica pelo contador
            return

        # print(f"[TensorBoardManager] Flushing {len(self._scalar_buffer)} scalars and {len(self._image_buffer)} images...") # Debug
        start_time = time.time()

        # Copia os buffers para liberar o lock rapidamente
        scalars_to_write = self._scalar_buffer.copy()
        images_to_write = self._image_buffer.copy()

        # Limpa os buffers originais e reseta o contador
        self._scalar_buffer.clear()
        self._image_buffer.clear()
        # START: Resetar contador total
        items_flushed = self._total_items_in_buffer
        self._total_items_in_buffer = 0
        # END: Resetar contador total

        # Libera o lock aqui, permitindo que novos dados sejam adicionados
        # enquanto os dados copiados são escritos no disco.
        # NOTA: Isso requer que o lock NÃO seja mantido durante as chamadas
        #       de escrita do SummaryWriter. Ajustaremos `flush()` e `close()`.

        # --- Processamento e escrita fora do lock ---
        try:
            # Processa e escreve escalares
            for tag, scalar_value, step in scalars_to_write:
                try:
                    self._writer.add_scalar(tag, scalar_value, step)
                except Exception as e:
                    warnings.warn(f"[TensorBoardManager] Erro ao escrever escalar '{tag}' (step {step}): {e}")

            # Processa e escreve imagens (já são np.ndarray)
            for tag, img_array, step, dataformats in images_to_write:
                try:
                    # START: Remoção da conversão de imagem (já feita)
                    self._writer.add_image(tag, img_array, step, dataformats=dataformats)
                    # END: Remoção da conversão de imagem (já feita)
                except Exception as e:
                    warnings.warn(f"[TensorBoardManager] Erro ao escrever imagem '{tag}' (step {step}): {e}")

            # Garante que os dados sejam enviados para o disco
            self._writer.flush()

        finally:
            # Atualiza o tempo do último flush, independentemente de sucesso/erro parcial
            # Fazemos isso fora do lock para refletir o tempo de conclusão da E/S.
            self._last_flush_time = time.time()
            # print(f"[TensorBoardManager] Flush de {items_flushed} itens concluído em {self._last_flush_time - start_time:.2f}s") # Debug

    def flush(self):
        """Força a escrita de todos os buffers pendentes no disco."""
        # START: Ajuste no flush manual
        # Adquire o lock, verifica condições (com force=True) que chamará _flush_buffers.
        # _flush_buffers fará a cópia/limpeza sob lock e a escrita fora do lock.
        with self._buffer_lock:
             # print("[TensorBoardManager] Flush manual solicitado.") # Debug
             # Força a verificação, que chamará _flush_buffers se houver itens.
             self._check_flush_conditions(force=True)
        # END: Ajuste no flush manual

    def close(self):
        """Escreve quaisquer dados restantes no buffer e fecha o SummaryWriter."""
        print("[TensorBoardManager] Fechando...")
        # START: Ajuste no close
        # Chama flush() para garantir que a lógica de flush seja executada
        # (adquirindo lock, copiando/limpando sob lock, escrevendo fora do lock).
        self.flush()
        # Espera um pouco pode ser útil se a escrita do flush ainda estiver ocorrendo
        # em background, mas _writer.close() deve lidar com isso.
        # time.sleep(0.1) # Geralmente não necessário

        # Fecha o writer subjacente. Isso também deve chamar flush internamente no writer.
        try:
            self._writer.close()
        except Exception as e:
             warnings.warn(f"[TensorBoardManager] Erro ao fechar o SummaryWriter: {e}")
        # END: Ajuste no close
        print("[TensorBoardManager] Fechado.")

    # --- Métodos adicionais do SummaryWriter (opcional) ---
    # Você pode adicionar wrappers para outros métodos se precisar deles,
    # por exemplo, add_histogram, add_graph, etc., seguindo o mesmo padrão.
    # Exemplo:
    # def add_histogram(self, tag: str, values: Any, global_step: Optional[int] = None, ...):
    #     with self._buffer_lock:
    #         # Adicione aos buffers apropriados ou chame self._writer diretamente
    #         # se o buffering não for desejado/necessário para este tipo.
    #         # Para histogramas, talvez seja melhor escrever diretamente devido ao tamanho.
    #         self._writer.add_histogram(tag, values, global_step, ...)

    # Permite acesso direto ao writer subjacente, se necessário (use com cuidado)
    @property
    def writer(self):
        return self._writer


# END: Adição Módulo TensorBoardManager
