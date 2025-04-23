# modules/util/TensorBoardManager.py
import time
from collections import deque
from threading import Lock, Thread
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Optional, Dict, Tuple, List


# START: Adição Módulo TensorBoardManager
class TensorBoardManager:
    """
    Um wrapper para torch.utils.tensorboard.SummaryWriter que acumula
    dados na memória e os escreve no disco de forma agrupada para
    reduzir a contenção de I/O.
    """

    def __init__(
        self, log_dir: str, flush_secs: int = 120, max_queue_size: int = 10000
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

        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self._log_dir = log_dir
        self._max_queue_size = max_queue_size

        # Buffers para diferentes tipos de dados
        self._scalar_buffer: List[Tuple[str, Any, int]] = []
        self._image_buffer: List[Tuple[str, Any, int, str]] = []
        # Adicione outros buffers conforme necessário (e.g., histogram, audio)

        self._buffer_lock = Lock()
        self._last_flush_time = time.time()

        print(f"[TensorBoardManager] Inicializado para o diretório: {log_dir}")

    def add_scalar(
        self, tag: str, scalar_value: Any, global_step: Optional[int] = None
    ):
        """Adiciona um escalar ao buffer."""
        # Garante que o valor seja um tipo numérico básico (float/int)
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.detach().cpu().item()

        with self._buffer_lock:
            self._scalar_buffer.append((tag, scalar_value, global_step))
            self._check_flush_needed(
                force=len(self._scalar_buffer) > self._max_queue_size * 0.8
            )  # Flush se perto do limite

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        dataformats="CHW",
    ):
        """Adiciona uma imagem ao buffer."""
        # Armazena o tensor diretamente para evitar cópias desnecessárias agora.
        # A conversão/processamento ocorrerá durante o flush.
        # Nota: Isso pode consumir RAM se muitas imagens grandes forem logadas.
        with self._buffer_lock:
            self._image_buffer.append((tag, img_tensor, global_step, dataformats))
            self._check_flush_needed(
                force=len(self._image_buffer) > self._max_queue_size * 0.2
            )  # Flush se perto do limite

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
        """Escreve o conteúdo dos buffers no disco usando o SummaryWriter."""
        if not self._scalar_buffer and not self._image_buffer:
            return

        # print(f"[TensorBoardManager] Flushing {len(self._scalar_buffer)} scalars and {len(self._image_buffer)} images...") # Debug
        start_time = time.time()

        # Processa e escreve escalares
        scalars_to_write = self._scalar_buffer.copy()
        self._scalar_buffer.clear()
        for tag, scalar_value, step in scalars_to_write:
            try:
                self._writer.add_scalar(tag, scalar_value, step)
            except Exception as e:
                print(f"[TensorBoardManager] Erro ao escrever escalar '{tag}': {e}")

        # Processa e escreve imagens
        images_to_write = self._image_buffer.copy()
        self._image_buffer.clear()
        for tag, img_tensor, step, dataformats in images_to_write:
            try:
                # Garante que o tensor esteja na CPU para add_image, se necessário
                if isinstance(img_tensor, torch.Tensor):
                    img_tensor = img_tensor.detach().cpu()
                self._writer.add_image(tag, img_tensor, step, dataformats=dataformats)
            except Exception as e:
                print(f"[TensorBoardManager] Erro ao escrever imagem '{tag}': {e}")
                # traceback.print_exc() # Descomente para mais detalhes do erro

        # Garante que os dados sejam enviados para o disco (importante!)
        self._writer.flush()
        self._last_flush_time = time.time()
        # print(f"[TensorBoardManager] Flush concluído em {time.time() - start_time:.2f}s") # Debug

    def flush(self):
        """Força a escrita de todos os buffers pendentes no disco."""
        with self._buffer_lock:
            self._flush_buffers()

    def close(self):
        """Escreve quaisquer dados restantes no buffer e fecha o SummaryWriter."""
        print("[TensorBoardManager] Fechando...")
        with self._buffer_lock:
            self._flush_buffers()  # Garante que tudo seja escrito antes de fechar
        self._writer.close()
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
