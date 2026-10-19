import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

LogFormat = Literal["text", "json"]

TEXT_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

# Стандартные атрибуты LogRecord: всё остальное пришло через extra=
_RECORD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Одна JSON-строка на запись: удобно разбирать в Loki (``| json``).

    Поля из ``extra=`` (event, endpoint, itemid, score, ...) попадают
    в JSON как есть.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {k: v for k, v in vars(record).items() if k not in _RECORD_ATTRS}
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logger(
    log_path: Path = Path("logs/app.log"), fmt: LogFormat = "text"
) -> None:
    """
    Настраивает корневой логгер: вывод в консоль и в файл (UTF-8).
    Вызывается один раз в точке входа (скрипт, сервис).

    Args:
        log_path: Путь к файлу, куда будут записываться логи.
        fmt: text — человекочитаемый, json — для сбора в Loki.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
