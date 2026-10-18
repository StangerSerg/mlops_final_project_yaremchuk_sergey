from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла.

    Args:
        path: Путь к файлу с данными.

    Returns:
        DataFrame с загруженными данными.
    """
    df = pd.read_csv(path)
    return df


def load_parquet_data(path: Path) -> pd.DataFrame:
    """
    Загружает данные из parquet-файла.

    Args:
        path: Путь к файлу с данными.

    Returns:
        DataFrame с загруженными данными.
    """
    df = pd.read_parquet(path)
    return df