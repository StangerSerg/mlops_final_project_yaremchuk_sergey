"""Скачивает артефакты сервиса (модель и признаки товаров) из S3.

Используется init-контейнером в Kubernetes: файлы кладутся в
LOCAL_MODELS_DIR / LOCAL_PROCESSED_DIR (общий emptyDir), откуда их
читает основной контейнер. В отличие от сервиса, здесь нет fallback:
если S3 не настроен или файла нет — выходим с ошибкой, и под не стартует.
"""

import argparse
import logging
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.settings import Settings, get_settings
from src.utils.storage import ArtifactStorage

logger = logging.getLogger(__name__)


def fetch_artifacts(settings: Settings, require_s3: bool = True) -> list[Path]:
    """Скачивает модель и item_features, возвращает локальные пути.

    Raises:
        RuntimeError: S3 не настроен, а require_s3=True.
        FileNotFoundError: артефакт не найден.
    """
    storage = ArtifactStorage(settings)
    if require_s3 and not storage.s3_enabled:
        raise RuntimeError("S3 не настроен: задайте S3_BUCKET и ключи доступа")
    config = load_config(settings.config_path)
    paths = [
        storage.fetch("models", config.model_file),
        storage.fetch("processed", config.item_features_file),
    ]
    for path in paths:
        logger.info("Готов артефакт %s (%d байт)", path, path.stat().st_size)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="не требовать S3 (взять локальные файлы, если они есть)",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logger(settings.log_path)
    fetch_artifacts(settings, require_s3=not args.allow_local)


if __name__ == "__main__":
    main()
