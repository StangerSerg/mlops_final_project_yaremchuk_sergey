import logging
from pathlib import Path
from typing import Any, Literal

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.settings import Settings

logger = logging.getLogger(__name__)

Area = Literal["raw", "processed", "models"]


class ArtifactStorage:
    """Хранилище артефактов: S3 с локальным fallback.

    Каждая «область» (raw / processed / models) соответствует каталогу
    ``S3_PREFIX/<dir>`` в бакете и локальному каталогу из настроек.
    Локальный каталог служит кэшем для скачанных файлов и fallback-ом,
    если S3 не настроен или недоступен.

    Args:
        settings: Настройки окружения.
        client: Готовый boto3 S3-клиент (например, из Airflow S3Hook,
            где ключи хранятся в Connection). Тогда для работы с S3
            достаточно S3_BUCKET, ключи в настройках не нужны.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._external_client = client is not None

    @property
    def s3_enabled(self) -> bool:
        if self._external_client:
            return bool(self.settings.s3_bucket)
        return self.settings.s3_enabled

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                region_name=self.settings.s3_region,
                aws_access_key_id=self.settings.aws_access_key_id.get_secret_value(),
                aws_secret_access_key=(
                    self.settings.aws_secret_access_key.get_secret_value()
                ),
            )
        return self._client

    def local_path(self, area: Area, name: str) -> Path:
        dirs = {
            "raw": self.settings.local_raw_dir,
            "processed": self.settings.local_processed_dir,
            "models": self.settings.local_models_dir,
        }
        return dirs[area] / name

    def s3_key(self, area: Area, name: str) -> str:
        if area == "models" and self.settings.model_s3_key:
            return self.settings.model_s3_key
        dirs = {
            "raw": self.settings.s3_raw_dir,
            "processed": self.settings.s3_processed_dir,
            "models": self.settings.s3_models_dir,
        }
        return f"{self.settings.s3_prefix.strip('/')}/{dirs[area]}/{name}"

    def fetch(self, area: Area, name: str) -> Path:
        """Возвращает локальный путь к артефакту, при необходимости скачав его.

        Если S3 включён, файл скачивается в локальный каталог области.
        При ошибке S3 используется уже лежащая локально копия (fallback).

        Raises:
            FileNotFoundError: артефакт не найден ни в S3, ни локально.
        """
        path = self.local_path(area, name)
        if self.s3_enabled:
            key = self.s3_key(area, name)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.settings.s3_bucket, key, str(path))
                logger.info(
                    "Скачан s3://%s/%s -> %s", self.settings.s3_bucket, key, path
                )
                return path
            except (BotoCoreError, ClientError) as e:
                logger.warning(
                    "Не удалось скачать s3://%s/%s (%r), пробуем локальный файл %s",
                    self.settings.s3_bucket,
                    key,
                    e,
                    path,
                )
        if not path.exists():
            raise FileNotFoundError(f"Артефакт не найден ни в S3, ни локально: {path}")
        logger.info("Используется локальный файл %s", path)
        return path

    def publish(self, area: Area, name: str) -> str:
        """Загружает локальный файл области в S3 (если S3 включён).

        Returns:
            URI, по которому лежит артефакт (s3://... или локальный путь).
        """
        path = self.local_path(area, name)
        if not self.s3_enabled:
            logger.info("S3 не настроен, артефакт остаётся локально: %s", path)
            return str(path)
        key = self.s3_key(area, name)
        self.client.upload_file(str(path), self.settings.s3_bucket, key)
        uri = f"s3://{self.settings.s3_bucket}/{key}"
        logger.info("Загружен %s -> %s", path, uri)
        return uri

    def write_parquet(self, df: pd.DataFrame, area: Area, name: str) -> str:
        path = self.local_path(area, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return self.publish(area, name)
