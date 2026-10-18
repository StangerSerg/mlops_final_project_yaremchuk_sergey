from pathlib import Path

from pydantic import ValidationError

from src.schemas.output_schemas import Config


def load_config() -> Config:
    """Загружает в память конфигурационный файл.

    Returns:
        dict: Словарь с настройками из config.json.
    """
    try:
        return Config.model_validate_json(
            Path("config/config.json").read_text(encoding="utf-8")
        )
    except ValidationError as e:
        raise SystemExit(f"Некорректный конфиг :\n{e}") from e


def get_data_paths(config: Config) -> tuple[Path, Path]:
    """Извлекает из конфигурации пути для загрузки сырых данных.

    Returns:
        tuple[Path, Path]: Кортеж путей до файла events и item_properties.
    """
    events_path = Path(config.events_path)
    item_props_path = Path(config.item_props_path)

    return events_path, item_props_path


def get_output_paths(config: Config) -> tuple[Path, Path]:
    """Извлекает из конфигурации пути для сохранения обработанных файлов.

    Returns:
        tuple[Path, Path]: Кортеж путей до файла с фичами и файла для обучения модели.
    """
    out_item_feats_path = Path(config.out_item_feats_path)
    out_train_path = get_training_data_path(config)

    return out_item_feats_path, out_train_path


def get_training_paths(config: Config) -> tuple[Path, Path]:
    """Извлекает из конфигурации путь до файла с данными для обучения
        и путь сохранения обученной модели.

    Returns:
        tuple[Path, Path]: Кортеж, путь сохранения обученной модели
         и путь до файла с данными для обучения модели.
    """
    model_path = Path(config.output_model_path)
    data_path = get_training_data_path(config)

    return model_path, data_path


def get_training_data_path(config: Config) -> Path:
    return Path(config.out_train_path)