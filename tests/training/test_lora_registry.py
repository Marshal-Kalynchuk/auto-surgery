from __future__ import annotations

from pathlib import Path

from auto_surgery.training.lora_registry import AdapterRecord, LoRARegistry


def test_lora_registry_append(tmp_path: Path) -> None:
    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    reg = LoRARegistry(root_uri)
    reg.append(AdapterRecord(adapter_hash="a1", base_model_hash="b1", eval_score=0.9))
    promoted = reg.promote_if_eval_ok(
        AdapterRecord(adapter_hash="a2", base_model_hash="b1", eval_score=0.5),
        min_score=0.8,
    )
    assert promoted.promoted is False
