"""Prepare deterministic forceps collision proxy mesh assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_asset_path

TARGET_FACE_COUNT = 500
TARGET_FACE_TOLERANCE_BAND = 0.10


def _load_mesh_module() -> Any:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dep
        raise RuntimeError(
            "trimesh is required for collision mesh prep. Install with: uv sync --group prep"
        ) from exc
    return trimesh


def _load_body_mesh(input_path: Path):
    trimesh = _load_mesh_module()
    mesh = trimesh.load_mesh(str(input_path))
    if mesh is None:
        raise RuntimeError(f"Could not load mesh from: {input_path}")
    return mesh


def _crop_shaft_tip(mesh, ratio: float = 0.2) -> Any:
    if not (0.0 < ratio < 1.0):
        raise ValueError("Crop ratio must be between 0 and 1 (exclusive).")

    vertices = mesh.vertices
    if vertices.size == 0:
        raise ValueError("Input mesh has no vertices.")

    z_values = vertices[:, 2]
    z_max = float(z_values.max())
    z_min = float(z_values.min())
    cutoff = z_max - ratio * (z_max - z_min)
    keep_face_mask = (z_values[mesh.faces[:, 0]] >= cutoff) & (
        z_values[mesh.faces[:, 1]] >= cutoff
    ) & (z_values[mesh.faces[:, 2]] >= cutoff)
    if not np.any(keep_face_mask):
        raise ValueError("Crop ratio removes all faces; adjust ratio before retrying.")
    return mesh.submesh([keep_face_mask], append=True, only_watertight=False)


def _decimate(mesh, target_face_count: int = TARGET_FACE_COUNT):
    min_faces, max_faces = _target_face_count_range(target_face_count)
    if mesh.faces.shape[0] < min_faces:
        raise ValueError(
            "Collision proxy mesh has too few faces after trimming and "
            "cannot meet target tolerance: "
            f"{mesh.faces.shape[0]} < {min_faces}."
        )
    if mesh.faces.shape[0] <= max_faces:
        return mesh
    decimate_fn = getattr(mesh, "simplify_quadratic_decimation", None)
    if decimate_fn is None:
        raise RuntimeError("Mesh decimation method unavailable in installed trimesh backend.")
    return decimate_fn(target_face_count)


def _mesh_signature(mesh) -> str:
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy(order="C")
    faces = np.asarray(mesh.faces, dtype=np.int64).copy(order="C")
    digest = hashlib.sha256()
    digest.update(vertices.shape[0].to_bytes(8, byteorder="little", signed=False))
    digest.update(faces.shape[0].to_bytes(8, byteorder="little", signed=False))
    digest.update(vertices.tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def _target_face_count_range(target_face_count: int) -> tuple[int, int]:
    if target_face_count <= 0:
        raise ValueError("Target face count must be positive.")
    tolerance = max(0.0, min(TARGET_FACE_TOLERANCE_BAND, 1.0))
    min_faces = max(1, int(target_face_count * (1.0 - tolerance)))
    max_faces = max(min_faces, int(target_face_count * (1.0 + tolerance)))
    return min_faces, max_faces


def _validate_deterministic(
    mesh, output_path: Path, target_face_count: int = TARGET_FACE_COUNT
) -> str:
    if mesh.vertices.shape[0] == 0:
        raise ValueError("Collision proxy has zero vertices.")
    if mesh.faces.shape[0] == 0:
        raise ValueError("Collision proxy has zero faces.")
    if mesh.faces.shape[1] != 3:
        raise ValueError("Collision proxy does not contain triangular faces.")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError("Collision proxy contains NaN or infinite vertices.")
    if not np.all(np.isfinite(mesh.faces)):
        raise ValueError("Collision proxy contains invalid face indices.")
    if np.any(mesh.area_faces <= 0):
        raise ValueError("Collision proxy contains degenerate or zero-area faces.")
    min_faces, max_faces = _target_face_count_range(target_face_count)
    if not (min_faces <= mesh.faces.shape[0] <= max_faces):
        raise ValueError(
            "Collision proxy face count is outside allowed tolerance band: "
            f"{mesh.faces.shape[0]} not in [{min_faces}, {max_faces}] "
            f"for target {target_face_count}."
        )
    if not np.isfinite(mesh.area):
        raise ValueError("Collision proxy area is not finite.")
    if hasattr(mesh, "is_watertight") and not bool(mesh.is_watertight):
        raise ValueError("Collision proxy is not watertight.")
    if hasattr(mesh, "is_winding_consistent") and not bool(mesh.is_winding_consistent):
        raise ValueError("Collision proxy has inconsistent winding.")
    if not output_path.suffix.lower() == ".obj":
        raise ValueError("Collision proxy output must be an OBJ file.")
    return _mesh_signature(mesh)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve_committed_hash_path(output_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(f"{output_path}.sha256")


def _read_committed_mesh_signature(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Committed mesh signature file is empty: {path}")
    is_hex = len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )
    if not is_hex:
        raise ValueError(
            "Committed mesh signature file does not contain a valid SHA256 hex digest: "
            f"{path}"
        )
    return value.lower()


def _verify_committed_mesh(mesh, committed_hash_path: Path) -> str:
    signature = _mesh_signature(mesh)
    expected = _read_committed_mesh_signature(committed_hash_path)
    if signature != expected:
        raise ValueError(
            "Collision mesh signature mismatch. "
            f"Expected {expected}, got {signature}."
        )
    return signature


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic collision proxy from DejaVu forceps body mesh. "
            "Writes assets/forceps/shaft_tip_collision.obj"
        )
    )
    parser.add_argument(
        "--body-mesh",
        default=None,
        help="Path to DejaVu body mesh (defaults to scenes/liver/data/dv_tool/body_uv.obj).",
    )
    parser.add_argument(
        "--output",
        default="assets/forceps/shaft_tip_collision.obj",
        help="Destination OBJ path.",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.2,
        help="Portion of shaft top range to remove before decimation.",
    )
    parser.add_argument(
        "--faces",
        type=int,
        default=TARGET_FACE_COUNT,
        help="Target triangle count after decimation.",
    )
    parser.add_argument(
        "--verify-committed",
        action="store_true",
        help="Verify mesh signature against the committed geometry hash file.",
    )
    parser.add_argument(
        "--committed-hash",
        default=None,
        help="Path to committed mesh signature file (defaults to <output>.sha256).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output path.",
    )
    parser.add_argument(
        "--print-hash",
        action="store_true",
        help="Print SHA256 hash of generated mesh.",
    )
    return parser.parse_args()


def _resolve_body_mesh_path(custom_path: str | None) -> Path:
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return resolve_dejavu_asset_path("scenes/liver/data/dv_tool/body_uv.obj")


def build_shaft_tip_collision(
    *,
    body_mesh_path: str | Path,
    output_path: str | Path,
    crop_ratio: float = 0.2,
    target_faces: int = TARGET_FACE_COUNT,
    force: bool = False,
) -> Path:
    body_path = Path(body_mesh_path).expanduser().resolve()
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_file():
        raise RuntimeError(f"Output path exists and is not a file: {output}")

    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}. Re-run with --force to overwrite.")
    mesh = _load_body_mesh(body_path)
    trimmed = _crop_shaft_tip(mesh, ratio=crop_ratio)
    decimated = _decimate(trimmed, target_face_count=target_faces)
    _validate_deterministic(decimated, output, target_face_count=target_faces)
    decimated.export(output)
    return output


def main() -> None:
    args = _parse_args()
    output = Path(args.output).expanduser().resolve()
    committed_hash_path = _resolve_committed_hash_path(output, explicit=args.committed_hash)
    if args.verify_committed:
        mesh = _load_body_mesh(output)
        signature = _verify_committed_mesh(mesh, committed_hash_path=committed_hash_path)
        print(f"mesh_signature={signature}")
        return

    output = build_shaft_tip_collision(
        body_mesh_path=_resolve_body_mesh_path(args.body_mesh),
        output_path=output,
        crop_ratio=args.crop_ratio,
        target_faces=args.faces,
        force=args.force,
    )
    if args.print_hash:
        print(_sha256(output))
        mesh = _load_body_mesh(output)
        print(f"mesh_signature={_mesh_signature(mesh)}")


if __name__ == "__main__":
    main()
