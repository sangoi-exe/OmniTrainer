import queue
import threading
from collections.abc import Iterable, Iterator
from contextlib import nullcontext, suppress

import torch


class PrefetchIterator:
    def __init__(self, iterable: Iterable, queue_size: int = 1, stop_poll_interval: float = 0.1):
        self._iterable = iterable
        self._queue_size = queue_size
        self._stop_poll_interval = stop_poll_interval
        self._item_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._idle_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._stream_context = None
        self._stop_sentinel = object()

    def __iter__(self) -> Iterator:
        self._start()
        return self

    def __next__(self):
        self._start()
        item = self._item_queue.get()
        if item is self._stop_sentinel:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        return item

    def _start(self):
        if self._thread is not None:
            return

        self._item_queue = queue.Queue(maxsize=self._queue_size)
        self._stop_event = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._stream_context = torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _put_or_stop(self, value) -> bool:
        while not self._stop_event.is_set():
            with suppress(queue.Full):
                self._item_queue.put(value, timeout=self._stop_poll_interval)
                return True
        return False

    def _produce(self):
        with self._stream_context:
            iterator = iter(self._iterable)
            while not self._stop_event.is_set():
                try:
                    self._idle_event.clear()
                    item = next(iterator)
                except StopIteration:
                    self._idle_event.set()
                    self._put_or_stop(self._stop_sentinel)
                    return
                except BaseException as error:
                    self._idle_event.set()
                    self._put_or_stop(error)
                    return

                self._idle_event.set()
                if not self._put_or_stop(item):
                    return

    def wait_until_idle(self):
        self._start()
        self._idle_event.wait()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self):
        if self._stop_event is None:
            return

        self._stop_event.set()
        if self._item_queue is not None:
            with suppress(queue.Empty):
                while True:
                    self._item_queue.get_nowait()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join()

    def __del__(self):
        self.close()
