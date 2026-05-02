"""Substrate runtime wiring point."""

from __future__ import annotations


class SubstrateRuntime:
    def __init__(self, model: object | None = None) -> None:
        self._model = model

    @property
    def model(self) -> object | None:
        return self._model
