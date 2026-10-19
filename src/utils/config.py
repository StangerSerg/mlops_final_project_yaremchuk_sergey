from pathlib import Path

from pydantic import ValidationError

from src.schemas.config import Config


def load_config(path: Path = Path("config/config.json")) -> Config:
    """Загружает и валидирует ML-конфиг пайплайна.

    Args:
        path: Путь к config.json.

    Returns:
        Провалидированный объект Config.
    """
    try:
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise SystemExit(f"Некорректный конфиг {path}:\n{e}") from e
