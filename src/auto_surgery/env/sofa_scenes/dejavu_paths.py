"""Shared DejaVu path resolution and scene-template helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[4]
_DEJAVU_ENV_VARS = ("AUTO_SURGERY_DEJAVU_ROOT",)

DEJAVU_ROOT_PLACEHOLDER = "${DEJAVU_ROOT}"
DEFAULT_BRAIN_FORCEPS_SCENE_TEMPLATE = (
    _PACKAGE_ROOT
    / "src"
    / "auto_surgery"
    / "env"
    / "sofa_scenes"
    / "brain_dejavu_episodic.scn.template"
)


def resolve_dejavu_root(*, override: str | Path | None = None) -> Path:
    """Resolve a DejaVu root path from explicit config and environment."""

    configured: list[str] = []
    if override is not None:
        configured.append(str(override).strip())
    for name in _DEJAVU_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            configured.append(value)

    candidates = tuple(Path(value).expanduser() for value in configured if value)
    if not candidates:
        raise RuntimeError(
            "AUTO_SURGERY_DEJAVU_ROOT is required for DejaVu scene/asset resolution. "
            "Set the environment variable and retry."
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "Configured DejaVu root(s) do not exist or are inaccessible. Checked: "
        + checked
    )


def resolve_dejavu_asset_path(
    relative_path: str | Path, *, root: str | Path | None = None
) -> Path:
    """Return an absolute DejaVu asset path."""

    return (resolve_dejavu_root(override=root) / str(relative_path)).resolve()


def render_dejavu_scene_template(
    template_path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> str:
    """Render a DejaVu scene template by replacing `${DEJAVU_ROOT}`."""

    template = Path(
        template_path
        if template_path is not None
        else DEFAULT_BRAIN_FORCEPS_SCENE_TEMPLATE
    ).expanduser()
    resolved_root = resolve_dejavu_root(override=root).as_posix()
    raw_xml = template.read_text(encoding="utf-8")
    rendered_xml = raw_xml.replace(DEJAVU_ROOT_PLACEHOLDER, resolved_root)
    if rendered_xml == raw_xml:
        return str(template)

    with tempfile.NamedTemporaryFile(
        suffix=".scn",
        prefix="auto-surgery-brain-dejavu-",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(rendered_xml)
        return handle.name


def resolve_brain_forceps_scene_path(
    scene_path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> str:
    """Return a scene path with DejaVu references concretized for runtime loading."""

    if scene_path is None:
        return render_dejavu_scene_template(root=root)

    resolved = Path(scene_path).expanduser().resolve()
    try:
        raw_xml = resolved.read_text(encoding="utf-8")
    except (OSError, TypeError):
        return str(resolved)
    if DEJAVU_ROOT_PLACEHOLDER in raw_xml:
        return render_dejavu_scene_template(resolved, root=root)
    return str(resolved)
