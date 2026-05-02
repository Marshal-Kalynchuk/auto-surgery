# auto-surgery

Phase 0 Python infrastructure: typed schemas, `Environment` protocol (sim/real), append-only Parquet logging with manifests, training dataset/checkpoint manifests, and Blackwell smoke tooling.

## Setup

```bash
uv sync --frozen --all-groups --extra train
```

## Checks

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run pyright
uv run pytest
```

## CLI

```bash
uv run auto-surgery smoke --skip-gpu   # CPU-only import checks
uv run auto-surgery smoke              # full CUDA smoke when torch+GPU present
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture.
