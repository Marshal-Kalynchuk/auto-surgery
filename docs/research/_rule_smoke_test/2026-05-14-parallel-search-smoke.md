# Rule smoke test: external research (parallel-style)

**Date:** 2026-05-14  
**Rule:** `.cursor/rules/external-references.mdc`  
**Intent:** Verify “check REFERENCES first → parallel external lookups → durable Markdown + JSON under `docs/research/`."

## Track A — SOFA (web search)

- **Query:** SOFA simulation framework official site sofa-framework.org  
- **Takeaway:** Official hub is `https://www.sofa-framework.org/` (about, docs, download). Framework is open-source, medical/biomechanics emphasis, C++ plugin architecture, LGPL open-core; scene graph / XML or Python scenes align with this repo’s SOFA usage.  
- **Note:** Cross-check install/runtime details against repo docs (`docs/SOFA_INSTALLATION.md`, etc.) before changing CI or local setup.

## Track B — DINOv2 (web search)

- **Query:** DINOv2 Meta AI paper arxiv 2023  
- **Takeaway:** arXiv **2304.07193** matches `docs/library/REFERENCES.md` (DINOv2 section). GitHub `facebookresearch/dinov2` and Hugging Face paper pages surfaced consistently with the canonical file.

## REFERENCES compliance

- Read `docs/library/REFERENCES.md` **before** broad search; DINOv2 URLs in that file were **not** contradicted by search results.  
- No new load-bearing dependency was introduced; **no edit** to `REFERENCES.md` required for this smoke test.

## Conclusion

The workflow is viable: two independent lookups, human-readable summary here, machine-readable companion `2026-05-14-parallel-search-smoke.json`.
