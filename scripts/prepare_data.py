import logging
from pathlib import Path

import pandas as pd

from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_events, preprocess_properties
from src.features.feature_engineering import prepare_features
from src.schemas.output_schemas import Config
from src.utils.config import get_data_paths, get_output_paths, load_config
from src.utils.logging import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def prepare() -> None:
    """
    Подготавливает данные для обучения модели.
    Последовательно выполняет:
      1. Загрузка сырых данных.
      2. Предобработка данных.
      3. Валидация данных.
      4. Собирает признаки для обучения модели ранжирования товаров.
      5. Сохраняет обработанные данные в формате .parquet
    """
    logger.info("Старт обучения модели")

    config: Config = load_config()
    logger.info("Настройки успешно загружены")

    window_hours: int = config.window_hours
    events_path: Path
    item_props_path: Path
    events_path, item_props_path = get_data_paths(config)
    logger.info(f"Пути успешно загружены: {events_path=}, {item_props_path=}")

    # Загрузка и предобработка
    raw_events: pd.DataFrame = load_data(events_path)
    logger.info(f"Данные о событиях успешно загружены,"
                f" размер: {raw_events.shape}")
    preprocessed_events = preprocess_events(raw_events)
    logger.info("Данные о событиях прошли предобработку.")

    raw_item_props = load_data(item_props_path)
    logger.info(f"Данные о свойствах товаров успешно загружены,"
                f" размер: {raw_item_props.shape}")
    preprocessed_properties = preprocess_properties(raw_item_props)
    logger.info("Данные о свойствах товаров прошли предобработку.")

    out_item_feats_path, out_train_path = get_output_paths(config)
    item_features, views = prepare_features(
        preprocessed_events,
        preprocessed_properties,
        window_hours
    )
    logger.info(f"Необходимые признаки успешно добавлены, "
                f"размер после обработки: {views.shape}.")

    item_features.to_parquet(out_item_feats_path, index=False)
    views.to_parquet(out_train_path, index=False)
    logger.info(f"Данные сохранены в формате .parquet по путям:"
                f"{out_item_feats_path=}, {out_train_path=}.")

if __name__ == "__main__":
    def main() -> None:
        prepare()

    main()