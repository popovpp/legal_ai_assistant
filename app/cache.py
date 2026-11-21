# app/cache.py
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Any


class InMemoryLRU:
    def __init__(self, max_size: int = 256):
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = Lock()
        self._max_size = max_size

    @staticmethod
    def make_key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(repr(p).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
