"""DAG препроцессинга: S3 raw_data -> признаки -> S3 processed_data.

Шаги:
    check_raw_data    — сырые файлы есть в бакете (размер в логе)
    extract           — скачать сырые CSV во временный каталог запуска
    transform         — предобработка и сборка признаков (код из src/)
    validate          — проверка обучающей выборки схемой pandera
    load              — выгрузка parquet в S3_PREFIX/processed_data
    cleanup           — удалить временный каталог (выполняется всегда)

Ключи S3 хранятся в Airflow Connection ``s3_storage`` (создаётся в
airflow-init из .env), бакет и префикс — в переменных окружения
S3_BUCKET / S3_PREFIX. Код проекта (src/, config/) смонтирован в
/opt/airflow/project и доступен через PYTHONPATH.
"""

import logging
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

logger = logging.getLogger(__name__)

S3_CONN_ID = "s3_storage"
WORK_DIR = Path(os.environ.get("PIPELINE_WORK_DIR", "/opt/airflow/work"))


def _run_dir() -> Path:
    run_id = str(get_current_context()["run_id"])
    return WORK_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def _storage() -> Any:
    """ArtifactStorage с клиентом из Airflow Connection и каталогами запуска."""
    from src.utils.settings import Settings
    from src.utils.storage import ArtifactStorage

    run_dir = _run_dir()
    settings = Settings(
        _env_file=None,
        local_raw_dir=run_dir / "raw",
        local_processed_dir=run_dir / "processed",
        local_models_dir=run_dir / "models",
    )
    storage = ArtifactStorage(
        settings, client=S3Hook(aws_conn_id=S3_CONN_ID).get_conn()
    )
    if not storage.s3_enabled:
        raise RuntimeError("S3_BUCKET не задан: DAG работает только с S3")
    return storage


def _config() -> Any:
    from src.utils.config import load_config
    from src.utils.settings import Settings

    return load_config(Settings(_env_file=None).config_path)


@dag(
    dag_id="preprocessing_data",
    description="S3 raw_data -> признаки и обучающая выборка -> S3 processed_data",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "sample": Param(
            None,
            type=["null", "integer"],
            minimum=1,
            description="Сколько строк читать из сырых CSV (пусто — все)",
        ),
    },
    tags=["recsys", "preprocessing"],
)
def preprocessing_data() -> None:
    @task
    def check_raw_data() -> list[str]:
        storage = _storage()
        config = _config()
        keys = []
        for name in (config.events_file, config.item_props_file):
            key = storage.s3_key("raw", name)
            head = storage.client.head_object(
                Bucket=storage.settings.s3_bucket, Key=key
            )
            logger.info(
                "Найден s3://%s/%s (%.1f MB)",
                storage.settings.s3_bucket,
                key,
                head["ContentLength"] / 2**20,
            )
            keys.append(key)
        return keys

    @task
    def extract() -> dict[str, str]:
        storage = _storage()
        config = _config()
        paths = {
            "events": str(storage.fetch("raw", config.events_file)),
            "item_props": str(storage.fetch("raw", config.item_props_file)),
        }
        logger.info("Сырые данные загружены из S3: %s", paths)
        return paths

    @task
    def transform(paths: dict[str, str]) -> dict[str, str]:
        from src.data.pipeline import build_datasets

        storage = _storage()
        config = _config()
        sample = get_current_context()["params"].get("sample")
        item_features, train = build_datasets(
            Path(paths["events"]),
            Path(paths["item_props"]),
            config.window_hours,
            sample,
        )
        out = {
            "item_features": storage.local_path("processed", config.item_features_file),
            "train": storage.local_path("processed", config.train_file),
        }
        out["train"].parent.mkdir(parents=True, exist_ok=True)
        item_features.to_parquet(out["item_features"], index=False)
        train.to_parquet(out["train"], index=False)
        logger.info(
            "Данные предобработаны: item_features %s, train %s, доля таргета %.5f",
            item_features.shape,
            train.shape,
            train["target"].mean(),
        )
        return {k: str(v) for k, v in out.items()}

    @task
    def validate(paths: dict[str, str]) -> dict[str, str]:
        import pandas as pd

        from src.data.validation import validate_data

        config = _config()
        train = pd.read_parquet(paths["train"])
        validate_data(train, config.required_columns)
        logger.info("Обучающая выборка прошла валидацию: %d строк", len(train))
        return paths

    @task
    def load(paths: dict[str, str]) -> list[str]:
        storage = _storage()
        config = _config()
        uris = [
            storage.publish("processed", config.item_features_file),
            storage.publish("processed", config.train_file),
        ]
        logger.info("Обработанные данные сохранены в S3: %s", uris)
        return uris

    @task(trigger_rule="all_done")
    def cleanup() -> None:
        shutil.rmtree(_run_dir(), ignore_errors=True)

    checked = check_raw_data()
    raw = extract()
    checked >> raw
    loaded = load(validate(transform(raw)))
    loaded >> cleanup()


preprocessing_data()
