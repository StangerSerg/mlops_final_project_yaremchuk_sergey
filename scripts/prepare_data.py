import argparse
import logging

from src.data.pipeline import build_datasets
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.settings import get_settings
from src.utils.storage import ArtifactStorage

logger = logging.getLogger(__name__)


def prepare(sample: int | None = None) -> None:
    """
    Подготавливает данные для обучения модели.
    Последовательно выполняет:
      1. Загрузка сырых данных (S3 raw_data или локальный data/raw).
      2. Предобработка данных и сборка признаков.
      3. Сохранение результатов в .parquet (S3 processed_data или data/prepared).

    Args:
        sample: Сколько строк читать из сырых файлов (None — все).
    """
    settings = get_settings()
    config = load_config(settings.config_path)
    storage = ArtifactStorage(settings)
    logger.info(
        "Старт подготовки данных (S3: %s, sample: %s)", storage.s3_enabled, sample
    )

    item_features, train = build_datasets(
        storage.fetch("raw", config.events_file),
        storage.fetch("raw", config.item_props_file),
        config.window_hours,
        sample,
    )

    feats_uri = storage.write_parquet(
        item_features, "processed", config.item_features_file
    )
    train_uri = storage.write_parquet(train, "processed", config.train_file)
    logger.info("Данные сохранены: %s, %s", feats_uri, train_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка признаков")
    parser.add_argument(
        "--sample", type=int, default=None, help="читать только N строк"
    )
    args = parser.parse_args()

    setup_logger(get_settings().log_path)
    prepare(sample=args.sample)


if __name__ == "__main__":
    main()
