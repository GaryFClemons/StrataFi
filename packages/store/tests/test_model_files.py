"""A trained model's files in the artifact store: each file a part of
the model's artifact, the manifest listing them with their SHA-256, and a read that refuses a
part that changed."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stratafi_core.components import ComponentError
from stratafi_core.training import BaseModelRef, ModelManifest
from stratafi_store.artifacts import LocalArtifactStore, artifact_key, artifact_part_key
from stratafi_store.training import ModelFiles, TrainedModel, put_model

WORKSPACE = uuid.UUID(int=1)
KEY = "d" * 64
BASE = BaseModelRef(component="embed/sentence-transformers", model="org/base", revision="main")


def trained(key: str = KEY) -> TrainedModel:
    now = datetime(2026, 11, 5, tzinfo=UTC)
    return TrainedModel(
        id=uuid.UUID(int=2),
        workspace_id=WORKSPACE,
        name="halcyon-minilm",
        version=1,
        kind="embedder",
        training_run_id=uuid.UUID(int=3),
        model_key=key,
        base=BASE.model_dump(mode="json"),
        card={},
        status="ready",
        withdrawn_by=None,
        withdrawn_at=None,
        created_by="dev",
        created_at=now,
        updated_at=now,
    )


def test_a_models_files_round_trip_with_their_checksums(tmp_path: Path) -> None:
    source = tmp_path / "model"
    (source / "1_Pooling").mkdir(parents=True)
    (source / "config.json").write_text('{"hidden_size": 8}')
    (source / "model.safetensors").write_bytes(b"\x00" * 64)
    (source / "1_Pooling" / "config.json").write_text("{}")
    paths = sorted(path for path in source.rglob("*") if path.is_file())
    store = LocalArtifactStore(tmp_path / "store")
    manifest, size, info = put_model(
        store, WORKSPACE, KEY, source, paths, kind="embedder", format="sentence-transformers",
        base=BASE,
    )  # fmt: skip
    assert [file.path for file in manifest.files] == [
        "1_Pooling/config.json",
        "config.json",
        "model.safetensors",
    ]
    assert [file.part for file in manifest.files] == ["f000", "f001", "f002"]
    assert size == sum(file.size for file in manifest.files) + info.size
    stored = ModelManifest.model_validate_json(store.get(artifact_key(WORKSPACE, KEY)))
    assert stored == manifest
    assert json.loads(store.get(artifact_key(WORKSPACE, KEY)))["schema"] == "stratafi.model.v1"

    files = ModelFiles(engine=None, store=store)  # type: ignore[arg-type]
    model = trained()
    assert files.read(model, manifest.files[2]) == b"\x00" * 64
    # A part that changed is refused, never loaded.
    store.put(artifact_part_key(WORKSPACE, KEY, "f002"), b"\x01" * 64)
    with pytest.raises(ComponentError) as error:
        files.read(model, manifest.files[2])
    assert error.value.code == "model_corrupt"
    with pytest.raises(ComponentError) as error:
        files.manifest(trained("e" * 64))
    assert error.value.code == "artifact_missing"
