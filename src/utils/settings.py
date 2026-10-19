from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Параметры окружения (S3, пути, сервис).

    Значения читаются из переменных окружения и файла ``.env``.
    Всё, что отличается между тестовым и боевым окружением
    (бакет, ключи, хосты), задаётся только здесь — не в коде.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- S3 ---
    s3_endpoint_url: str = "https://storage.yandexcloud.net"
    s3_region: str = "ru-central1"
    s3_bucket: str = ""
    s3_prefix: str = "yaremchuk_sergey_project"
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")

    # Подкаталоги внутри S3_PREFIX (структура задана в задании)
    s3_raw_dir: str = "raw_data"
    s3_processed_dir: str = "processed_data"
    s3_models_dir: str = "models"
    # Полный ключ модели в бакете, если она лежит не в S3_PREFIX/models
    # (например, лучшая модель из пайплайна ClearML).
    model_s3_key: str = ""

    # --- Локальные каталоги (кэш S3 и fallback, если S3 недоступен) ---
    local_raw_dir: Path = Path("data/raw")
    local_processed_dir: Path = Path("data/prepared")
    local_models_dir: Path = Path("models")

    # --- Сервис ---
    # имя и версия модели в V2 Inference Protocol (/v2/models/<name>)
    model_name: str = "ranker"
    model_version: str = "1"
    app_host: str = "0.0.0.0"
    app_port: int = Field(32000, ge=1, le=65535)
    config_path: Path = Path("config/config.json")
    log_path: Path = Path("logs/app.log")
    # json — для сбора логов в Loki (docker-compose на ВМ, k8s)
    log_format: Literal["text", "json"] = "text"

    @property
    def s3_enabled(self) -> bool:
        """S3 используется, только если заданы бакет и оба ключа."""
        return bool(
            self.s3_bucket
            and self.aws_access_key_id.get_secret_value()
            and self.aws_secret_access_key.get_secret_value()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
