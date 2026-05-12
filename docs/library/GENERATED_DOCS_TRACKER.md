# Generated docs/resources drift tracker

This file tracks links and generated resources that must remain synchronized across
documentation, scripts, and runtime helper modules.

## Canonical references

| Resource | Canonical location | Current references |
| --- | --- | --- |
| Architecture specification | `docs/library/ARCHITECTURE.md` | `README.md`, `docs/library/ARCHITECTURE_THESIS.md`, `docs/library/REFERENCES.md`, `docs/library/GRAPHITI_SCHEMA.md` |
| Architecture thesis | `docs/library/ARCHITECTURE_THESIS.md` | `docs/library/REFERENCES.md` |
| External references index | `docs/library/REFERENCES.md` | `docs/library/ARCHITECTURE_THESIS.md`, `docs/library/GRAPHITI_SCHEMA.md` |
| Scene template | `src/auto_surgery/env/sofa_scenes/brain_dejavu_forceps_poc.scn` | `auto_surgery.env.sofa_scenes.dejavu_paths.resolve_brain_forceps_scene_path`, `docs/SOFA_INSTALLATION.md`, `src/auto_surgery/recording/brain_forceps.py`, `src/auto_surgery/training/sofa_smoke.py` |

## Drift checks (manual)

- Run a repo-wide search after edits:
  - `rg "docs/ARCHITECTURE\\.md|docs/REFERENCES\\.md|docs/library/ARCHITECTURE.md|docs/library/REFERENCES.md"`
  - `rg "brain_dejavu_forceps_poc.scn|AUTO_SURGERY_DEJAVU_ROOT"`
- If a canonical location changes, update every reference above before merging.
- Keep generated variants (for example scene templates) in code-generation helpers
  (`src/auto_surgery/env/sofa_scenes/dejavu_paths.py`) so call sites never hardcode
  absolute paths.
