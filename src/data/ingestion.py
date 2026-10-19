from pathlib import Path

import pandas as pd


def load_data(
    path: Path,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла.

    Args:
        path: Путь к файлу с данными.
        usecols: Какие колонки читать (экономит память на больших файлах).
        nrows: Сколько строк читать (для быстрых прогонов на сэмпле).

    Returns:
        DataFrame с загруженными данными.
    """
    return pd.read_csv(path, usecols=usecols, nrows=nrows)


def load_parquet_data(path: Path) -> pd.DataFrame:
    """
    Загружает данные из parquet-файла.

    Args:
        path: Путь к файлу с данными.

    Returns:
        DataFrame с загруженными данными.
    """
    return pd.read_parquet(path)
