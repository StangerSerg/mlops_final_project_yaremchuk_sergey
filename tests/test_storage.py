from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from pydantic import SecretStr

from src.utils.settings import Settings
from src.utils.storage import ArtifactStorage


def _settings(tmp_path: Path, s3: bool) -> Settings:
    return Settings(
        _env_file=None,
        s3_bucket="test-bucket" if s3 else "",
        s3_prefix="student_project",
        aws_access_key_id=SecretStr("key" if s3 else ""),
        aws_secret_access_key=SecretStr("secret" if s3 else ""),
        local_raw_dir=tmp_path / "raw",
        local_processed_dir=tmp_path / "prepared",
        local_models_dir=tmp_path / "models",
    )


def _client_error() -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")


# ---------- settings ----------


def test_s3_disabled_without_bucket(tmp_path: Path) -> None:
    assert _settings(tmp_path, s3=False).s3_enabled is False


def test_s3_enabled_with_bucket_and_keys(tmp_path: Path) -> None:
    assert _settings(tmp_path, s3=True).s3_enabled is True


# ---------- keys / paths ----------


def test_s3_key_uses_prefix_and_area_dir(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=True), client=MagicMock())

    assert storage.s3_key("raw", "events.csv") == "student_project/raw_data/events.csv"


def test_s3_key_model_override(tmp_path: Path) -> None:
    settings = _settings(tmp_path, s3=True)
    settings.model_s3_key = "pipelines/best/ranker.pkl"
    storage = ArtifactStorage(settings, client=MagicMock())

    assert storage.s3_key("models", "ranker.pkl") == "pipelines/best/ranker.pkl"


def test_local_path_points_to_area_dir(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=False))

    assert storage.local_path("raw", "events.csv") == tmp_path / "raw" / "events.csv"


# ---------- fetch ----------


def test_fetch_local_returns_existing_file(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=False))
    path = storage.local_path("raw", "events.csv")
    path.parent.mkdir(parents=True)
    path.write_text("a\n1\n")

    assert storage.fetch("raw", "events.csv") == path


def test_fetch_local_missing_file_raises(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=False))

    with pytest.raises(FileNotFoundError):
        storage.fetch("raw", "events.csv")


def test_fetch_s3_downloads_to_local_dir(tmp_path: Path) -> None:
    client = MagicMock()
    storage = ArtifactStorage(_settings(tmp_path, s3=True), client=client)

    result = storage.fetch("raw", "events.csv")

    client.download_file.assert_called_once_with(
        "test-bucket",
        "student_project/raw_data/events.csv",
        str(tmp_path / "raw" / "events.csv"),
    )
    assert result == tmp_path / "raw" / "events.csv"


def test_fetch_s3_error_falls_back_to_local(tmp_path: Path) -> None:
    client = MagicMock()
    client.download_file.side_effect = _client_error()
    storage = ArtifactStorage(_settings(tmp_path, s3=True), client=client)
    path = storage.local_path("raw", "events.csv")
    path.parent.mkdir(parents=True)
    path.write_text("a\n1\n")

    assert storage.fetch("raw", "events.csv") == path


def test_fetch_s3_error_without_local_raises(tmp_path: Path) -> None:
    client = MagicMock()
    client.download_file.side_effect = _client_error()
    storage = ArtifactStorage(_settings(tmp_path, s3=True), client=client)

    with pytest.raises(FileNotFoundError):
        storage.fetch("raw", "events.csv")


# ---------- publish / write ----------


def test_publish_without_s3_returns_local_path(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=False))

    uri = storage.publish("models", "ranker.pkl")

    assert uri == str(tmp_path / "models" / "ranker.pkl")


def test_publish_with_s3_uploads_file(tmp_path: Path) -> None:
    client = MagicMock()
    storage = ArtifactStorage(_settings(tmp_path, s3=True), client=client)

    uri = storage.publish("models", "ranker.pkl")

    client.upload_file.assert_called_once_with(
        str(tmp_path / "models" / "ranker.pkl"),
        "test-bucket",
        "student_project/models/ranker.pkl",
    )
    assert uri == "s3://test-bucket/student_project/models/ranker.pkl"


def test_write_parquet_saves_local_file(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=False))
    df = pd.DataFrame({"a": [1, 2]})

    storage.write_parquet(df, "processed", "out.parquet")

    saved = pd.read_parquet(tmp_path / "prepared" / "out.parquet")
    pd.testing.assert_frame_equal(saved, df)


def test_external_client_enables_s3_without_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path, s3=False)
    settings.s3_bucket = "test-bucket"

    storage = ArtifactStorage(settings, client=MagicMock())

    assert storage.s3_enabled is True


def test_external_client_without_bucket_keeps_s3_disabled(tmp_path: Path) -> None:
    storage = ArtifactStorage(_settings(tmp_path, s3=False), client=MagicMock())

    assert storage.s3_enabled is False
