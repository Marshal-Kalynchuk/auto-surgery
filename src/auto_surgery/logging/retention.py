"""Retention tier helpers — durations are operational policy, not encoded here."""

from __future__ import annotations

from auto_surgery.schemas.manifests import RetentionTier

_RETENTION_DESCRIPTIONS: dict[RetentionTier, str] = {
    RetentionTier.RAW_SENSORS: (
        "High-volume multimodal archives; shortest PHI exposure window where applicable."
    ),
    RetentionTier.DERIVED_FEATURES: "Intermediate tensors / embeddings before curation.",
    RetentionTier.CURATED_TRAINING: "Training-ready slices referenced by DatasetManifest.",
    RetentionTier.CHECKPOINTS: "Model checkpoints with CheckpointManifest provenance.",
    RetentionTier.AUDIT_ONLY: "Minimal records required for regulatory audit.",
}


def describe_tier(tier: RetentionTier) -> str:
    return _RETENTION_DESCRIPTIONS[tier]
