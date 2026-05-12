from __future__ import annotations

import json
from pathlib import Path

from auto_surgery.logging.writer import frames_to_table, table_to_frames
from auto_surgery.schemas.logging import LoggedFrame


def test_logged_frame_roundtrip_via_parquet() -> None:
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "golden_logged_frame.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    frame = LoggedFrame.model_validate(raw)
    table = frames_to_table([frame])
    restored = table_to_frames(table)[0]
    assert restored.model_dump() == frame.model_dump()


def test_golden_fixture_loads() -> None:
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "golden_logged_frame.json"
    lf = LoggedFrame.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert lf.frame_index == 0
    assert lf.sensor_payload.tool.in_contact in (True, False)
