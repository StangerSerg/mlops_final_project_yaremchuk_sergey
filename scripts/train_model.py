import logging
from pathlib import Path

import joblib
import pandas as pd

from src.data.ingestion import load_parquet_data
from src.data.validation import validate_data
from src.models.evaluation import evaluate_model
from src.models.training import split_to_train_test, train_model
from src.schemas.output_schemas import Config
from src.utils.config import get_training_paths, load_config
from src.utils.logging import setup_logger


def train() -> None:
    """
    Обучает модель машинного обучения на основе конфигурации.
    Последовательно выполняет:
      1. Загрузка данных
      2. Обучение модели
      3. Сохранение модели
      4. Оценка качества модели
    """
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Старт обучения модели")

    # загрузка настроек
    config: Config = load_config()
    logger.info("Настройки успешно загружены")

    # получение путей
    data_path: Path
    model_path: Path
    model_path, data_path = get_training_paths(config)
    logger.info(f"Пути успешно загружены: {data_path=}, {model_path=}")

    # загрузка данных
    df = load_parquet_data(data_path)
    logger.info(f"Данные успешно загружены, размер: {df.shape}")

    # Валидация данных
    required_columns: list[str] = config.required_columns
    validate_data(df, required_columns)
    logger.info("Данные прошли валидацию")

    # создаем выборки для обучения и проверки
    features = config.features
    X: pd.DataFrame = df[features]
    y: pd.Series = df[config.target]
    x_train, x_test, y_train, y_test = split_to_train_test(X, y, config.split_params)

    # обучение и оценка модели
    model = train_model(x_train, y_train, config.model_params)
    metrics = evaluate_model(model, x_test, y_test)
    logger.info("Модель обучена")
    if metrics and isinstance(metrics, dict):
        for k, v in metrics.items():
            logger.info(f"{k:>10}: {v:.6f}")
    else:
        logger.info(f"Метрики получены неверно - {metrics}")

    # сохранение модели
    joblib.dump(model, model_path)
    logger.info(f"Модель сохранена по пути: {model_path}")

    logger.info("Обучение завершено успешно")


if __name__ == "__main__":
    train()