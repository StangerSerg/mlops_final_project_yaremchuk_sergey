from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from scripts.fetch_artifacts import fetch_artifacts
from src.utils.settings import Settings


def _settings(tmp_path: Path, s3: bool) -> Settings:
    return Settings(
        _env_file=None,
        s3_bucket="bucket" if s3 else "",
        aws_access_key_id=SecretStr("key" if s3 else ""),
        aws_secret_access_key=SecretStr("secret" if s3 else ""),
        local_processed_dir=tmp_path / "prepared",
        local_models_dir=tmp_path / "models",
    )


def test_fetch_artifacts_requires_s3(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        fetch_artifacts(_settings(tmp_path, s3=False))


def test_fetch_artifacts_allow_local_uses_existing_files(tmp_path: Path) -> None:
    for path in (
        tmp_path / "models/ranker.pkl",
        tmp_path / "prepared/item_features.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    paths = fetch_artifacts(_settings(tmp_path, s3=False), require_s3=False)

    assert [p.name for p in paths] == ["ranker.pkl", "item_features.parquet"]


def test_fetch_artifacts_downloads_model_and_features_from_s3(tmp_path: Path) -> None:
    client = MagicMock()
    client.download_file.side_effect = lambda bucket, key, dst: Path(dst).write_bytes(
        b"x"
    )

    with patch("src.utils.storage.boto3.client", return_value=client):
        fetch_artifacts(_settings(tmp_path, s3=True))

    keys = [call.args[1] for call in client.download_file.call_args_list]
    assert keys == [
        "yaremchuk_sergey_project/models/ranker.pkl",
        "yaremchuk_sergey_project/processed_data/item_features.parquet",
    ]
