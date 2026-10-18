import logging
from pathlib import Path


def setup_logger(log_path: Path = Path("logs/app.log")) -> None:
    """
    Настраивает логгер с записью в файл.
    Args:
        log_path: Путь к файлу, куда будут записываться логи.
    """
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )