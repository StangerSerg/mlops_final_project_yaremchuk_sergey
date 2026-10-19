import logging

import joblib

from src.data.ingestion import load_parquet_data
from src.data.validation import validate_data
from src.models.evaluation import evaluate_model
from src.models.training import split_to_train_test, train_model
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.settings import get_settings
from src.utils.storage import ArtifactStorage

logger = logging.getLogger(__name__)


def train() -> None:
    """
    Обучает модель на подготовленных данных.
    Последовательно выполняет:
      1. Загрузка данных (S3 processed_data или локальный data/prepared)
      2. Валидация данных
      3. Обучение и оценка модели
      4. Сохранение модели (S3 models или локальный models/)
    """
    settings = get_settings()
    config = load_config(settings.config_path)
    storage = ArtifactStorage(settings)
    logger.info("Старт обучения модели (S3: %s)", storage.s3_enabled)

    df = load_parquet_data(storage.fetch("processed", config.train_file))
    logger.info("Данные загружены, размер: %s", df.shape)

    validate_data(df, config.required_columns)
    logger.info("Данные прошли валидацию")

    x = df[config.features]
    y = df[config.target]
    x_train, x_test, y_train, y_test = split_to_train_test(x, y, config.split_params)

    model = train_model(x_train, y_train, config.model_params)
    metrics = evaluate_model(model, x_test, y_test)
    logger.info("Модель обучена")
    for name, value in metrics.items():
        logger.info("%10s: %.6f", name, value)

    model_path = storage.local_path("models", config.model_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    uri = storage.publish("models", config.model_file)
    logger.info("Модель сохранена: %s", uri)


def main() -> None:
    setup_logger(get_settings().log_path)
    train()


if __name__ == "__main__":
    main()
