"""Append-only Parquet session writer + session manifest."""

from __future__ import annotations

import io
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from auto_surgery.logging import checksums, storage
from auto_surgery.schemas.logging import LOGGED_FRAME_SCHEMA_VERSION, LoggedFrame
from auto_surgery.schemas.manifests import (
    SESSION_MANIFEST_SCHEMA_VERSION,
    DataClassification,
    RetentionTier,
    RunMetadata,
    SessionManifest,
)


def frames_to_table(frames: list[LoggedFrame]) -> pa.Table:
    """Serialize frames as JSON payloads for robust nested round-trips."""

    return pa.table(
        {
            "frame_index": pa.array([f.frame_index for f in frames], type=pa.int64()),
            "timestamp_ns": pa.array([f.timestamp_ns for f in frames], type=pa.int64()),
            "payload_json": pa.array([f.model_dump_json() for f in frames], type=pa.string()),
        }
    )


def table_to_frames(table: pa.Table) -> list[LoggedFrame]:
    payload_col = table.column("payload_json")
    return [LoggedFrame.model_validate_json(payload_col[i].as_py()) for i in range(table.num_rows)]


class SessionWriter:
    """Buffers frames and writes sealed Parquet segments (immutable after write)."""

    def __init__(
        self,
        storage_root: str,
        case_id: str,
        session_id: str,
        *,
        capture_rig_id: str,
        clock_source: str,
        software_git_sha: str,
        data_classification: DataClassification,
        retention_tier: RetentionTier = RetentionTier.RAW_SENSORS,
        sensor_list: list[str] | None = None,
        segment_max_frames: int = 256,
        fs_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._fs, self._root = fsspec.url_to_fs(storage_root, **(fs_kwargs or {}))
        self.case_id = case_id
        self.session_id = session_id
        self.capture_rig_id = capture_rig_id
        self.clock_source = clock_source
        self.software_git_sha = software_git_sha
        self.data_classification = data_classification
        self.retention_tier = retention_tier
        self.sensor_list = sensor_list or []
        self.segment_max_frames = segment_max_frames
        self._buffer: list[LoggedFrame] = []
        self._segment_index = 0
        self._checksums: dict[str, str] = {}
        self._prefix = storage.session_prefix(case_id, session_id)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for rel in (
            storage.segments_dir(self.case_id, self.session_id),
            storage.blobs_dir(self.case_id, self.session_id),
            "catalog",
        ):
            self._fs.makedirs(f"{self._root}/{rel}".rstrip("/"), exist_ok=True)

    def write_blob(self, relative_under_blobs: str, data: bytes) -> str:
        """Write raw bytes under this session's `blobs/` tree.

        Returns a path relative to the storage root.
        """

        blob_root = storage.blobs_dir(self.case_id, self.session_id)
        rel = f"{blob_root}/{relative_under_blobs.lstrip('/')}"
        full = f"{self._root}/{rel}"
        parent = full.rsplit("/", 1)[0]
        self._fs.makedirs(parent, exist_ok=True)
        self._fs.pipe_file(full, data)
        return rel

    def write_frame(self, frame: LoggedFrame) -> None:
        self._buffer.append(frame)
        if len(self._buffer) >= self.segment_max_frames:
            self._flush_segment()

    def _flush_segment(self) -> None:
        if not self._buffer:
            return
        rel = storage.segment_path(self.case_id, self.session_id, self._segment_index)
        full = f"{self._root}/{rel}"
        table = frames_to_table(self._buffer)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        data = buf.getvalue()
        checksum = checksums.sha256_bytes(data)
        self._fs.pipe_file(full, data)
        self._checksums[rel] = checksum
        self._segment_index += 1
        self._buffer.clear()

    def finalize(self, *, run_metadata: RunMetadata | None = None) -> SessionManifest:
        """Seal remaining frames and write `session_manifest.json`."""

        self._flush_segment()
        if run_metadata is not None:
            rel = storage.run_metadata_path(self.case_id, self.session_id)
            full = f"{self._root}/{rel}"
            parent = full.rsplit("/", 1)[0]
            self._fs.makedirs(parent, exist_ok=True)
            payload = run_metadata.model_dump_json(indent=2).encode("utf-8")
            self._fs.pipe_file(full, payload)
            self._checksums[rel] = checksums.sha256_bytes(payload)
        manifest = SessionManifest(
            schema_version=SESSION_MANIFEST_SCHEMA_VERSION,
            session_id=self.session_id,
            case_id=self.case_id,
            capture_rig_id=self.capture_rig_id,
            clock_source=self.clock_source,
            software_git_sha=self.software_git_sha,
            logged_frame_schema_version=LOGGED_FRAME_SCHEMA_VERSION,
            sensor_list=list(self.sensor_list),
            data_classification=self.data_classification,
            retention_tier=self.retention_tier,
            partition_checksums=dict(self._checksums),
        )
        man_path = f"{self._root}/{storage.session_manifest_path(self.case_id, self.session_id)}"
        payload = manifest.model_dump_json(indent=2).encode("utf-8")
        self._fs.pipe_file(man_path, payload)
        self._checksums[storage.session_manifest_path(self.case_id, self.session_id)] = (
            checksums.sha256_bytes(payload)
        )
        return manifest


def load_session_manifest(
    storage_root: str, case_id: str, session_id: str, **fs_kw: Any
) -> SessionManifest:
    fs, root = fsspec.url_to_fs(storage_root, **fs_kw)
    path = f"{root}/{storage.session_manifest_path(case_id, session_id)}"
    with fs.open(path, "rb") as f:
        raw = f.read()
    return SessionManifest.model_validate_json(raw.decode("utf-8"))


def load_segment_frames(
    storage_root: str, case_id: str, session_id: str, segment_index: int, **fs_kw: Any
) -> list[LoggedFrame]:
    fs, root = fsspec.url_to_fs(storage_root, **fs_kw)
    rel = storage.segment_path(case_id, session_id, segment_index)
    with fs.open(f"{root}/{rel}", "rb") as f:
        table = pq.read_table(f)
    return table_to_frames(table)


def session_manifest_to_json(manifest: SessionManifest) -> str:
    return manifest.model_dump_json(indent=2)


def count_segments(storage_root: str, case_id: str, session_id: str, **fs_kw: Any) -> int:
    """Return number of Parquet segment files on disk."""

    fs, root = fsspec.url_to_fs(storage_root, **fs_kw)
    seg_dir = f"{root}/{storage.segments_dir(case_id, session_id)}"
    if not fs.exists(seg_dir):
        return 0
    names = fs.ls(seg_dir, detail=False)
    return sum(1 for n in names if str(n).endswith(".parquet"))
