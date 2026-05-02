"""Bootstrap sanity loops M1–M3."""

from __future__ import annotations


def run_m1_tiny_overfit(*, steps: int = 300, device: str | None = None) -> float:
    """M1: overfit a tiny synthetic slice — validates wiring on Blackwell."""

    try:
        import torch
    except ImportError as e:
        raise ImportError("M1 requires torch (train extra).") from e
    from auto_surgery.models.substrate import build_tiny_substrate_mlp

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = build_tiny_substrate_mlp(16, 8).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    x = torch.randn(8, 16, device=dev)
    target = torch.randn(8, 8, device=dev)
    loss_val = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = net(x)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().cpu())
    return loss_val


def run_m2_contrastive_stub(*, steps: int = 100, device: str | None = None) -> float:
    """M2: tiny contrastive-style loss between two augmented views."""

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError("M2 requires torch (train extra).") from e
    from auto_surgery.models.embeddings import build_tiny_embedding_mlp

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = build_tiny_embedding_mlp(32, 16).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    a = torch.randn(16, 32, device=dev)
    b = a + 0.01 * torch.randn_like(a)
    loss_val = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        za = F.normalize(net(a), dim=-1)
        zb = F.normalize(net(b), dim=-1)
        loss = -torch.mean(torch.sum(za * zb, dim=-1))
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().cpu())
    return loss_val


def run_m3_imitation_stub_from_frames(
    frame_vectors: list[list[float]],
    action_targets: list[list[float]],
    *,
    steps: int = 400,
    device: str | None = None,
) -> float:
    """M3: shallow imitation mapping stacked frame features -> actions."""

    try:
        import torch
    except ImportError as e:
        raise ImportError("M3 requires torch (train extra).") from e
    from auto_surgery.models.substrate import build_tiny_substrate_mlp

    if len(frame_vectors) != len(action_targets) or not frame_vectors:
        raise ValueError("frame_vectors and action_targets must be non-empty and aligned.")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    obs_dim = len(frame_vectors[0])
    act_dim = len(action_targets[0])
    net = build_tiny_substrate_mlp(obs_dim, act_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    x = torch.tensor(frame_vectors, dtype=torch.float32, device=dev)
    y = torch.tensor(action_targets, dtype=torch.float32, device=dev)
    loss_val = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = net(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().cpu())
    return loss_val
