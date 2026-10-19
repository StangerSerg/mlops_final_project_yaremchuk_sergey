import logging
from pathlib import Path

import pandas as pd

from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_events, preprocess_properties
from src.features.feature_engineering import prepare_features

logger = logging.getLogger(__name__)

ITEM_PROPS_COLUMNS = ["itemid", "property", "value"]


def build_datasets(
    events_path: Path,
    item_props_path: Path,
    window_hours: int,
    sample: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сырые CSV -> признаки товаров и обучающая выборка.

    Общая точка для скрипта scripts/prepare_data.py и Airflow-DAG.

    Args:
        events_path: Локальный путь к events.csv.
        item_props_path: Локальный путь к item_properties_*.csv.
        window_hours: Окно (часы) для таргета «покупка после просмотра».
        sample: Сколько строк читать из сырых файлов (None — все).

    Returns:
        (item_features, train) — признаки товаров и обучающая выборка.
    """
    raw_events = load_data(events_path, nrows=sample)
    logger.info("Загружены события, размер: %s", raw_events.shape)
    events = preprocess_events(raw_events)

    raw_props = load_data(item_props_path, usecols=ITEM_PROPS_COLUMNS, nrows=sample)
    logger.info("Загружены свойства товаров, размер: %s", raw_props.shape)
    item_props = preprocess_properties(raw_props)

    item_features, train = prepare_features(events, item_props, window_hours)
    logger.info(
        "Признаки собраны: товаров %d, обучающая выборка %s, доля таргета %.5f",
        len(item_features),
        train.shape,
        float(train["target"].mean()) if len(train) else 0.0,
    )
    return item_features, train
