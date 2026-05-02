"""Checksum helpers for immutable shards."""

from __future__ import annotations

import hashlib
from typing import BinaryIO


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()
