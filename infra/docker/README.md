# Training container (Blackwell-oriented)

The reference training image installs PyTorch with CUDA support compatible with **NVIDIA Blackwell** GPUs. Because driver/CUDA/PyTorch matrices change frequently, **pin exact digests** after you validate smoke tests on hardware.

## Build

```bash
docker build -f infra/docker/Dockerfile.train -t auto-surgery-train:local .
```

## Smoke gate (blocking)

Inside the container:

```bash
auto-surgery smoke
```

On developer laptops without GPUs:

```bash
auto-surgery smoke --skip-gpu
```

## Compatibility notes

- Verify **GPU SKU**, **driver branch**, **CUDA userland**, and **PyTorch wheel** against NVIDIA’s Blackwell compatibility guide before freezing production images.
- Record the digest + versions in your internal infra registry; rebuild when upgrading CUDA minor versions.
