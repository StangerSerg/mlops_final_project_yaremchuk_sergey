import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingestion import load_parquet_data
from src.data.validation import validate_data
from src.schemas.config import Config, ModelParams, SplitParams
from src.utils.storage import ArtifactStorage

# Метрика выбора лучшей модели: положительных примеров очень мало,
# поэтому PR-AUC (average precision), а не ROC-AUC/accuracy.
SELECTION_METRIC = "pr_auc"

Splits = tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]


def load_training_frame(
    storage: ArtifactStorage, config: Config
) -> tuple[pd.DataFrame, pd.Series]:
    """Загружает обучающую выборку (processed_data), валидирует её и
    возвращает матрицу признаков и таргет."""
    df = load_parquet_data(storage.fetch("processed", config.train_file))
    validate_data(df, config.required_columns)
    return df[config.features], df[config.target]


def split_train_valid_test(
    x: pd.DataFrame, y: pd.Series, params: SplitParams
) -> Splits:
    """Стратифицированное разбиение на train / valid / test.

    test — отложенная выборка для финальной оценки лучшей модели,
    valid — для early stopping и сравнения кандидатов.

    Returns:
        (x_train, x_valid, x_test, y_train, y_valid, y_test).
    """
    x_rest, x_test, y_rest, y_test = train_test_split(
        x,
        y,
        test_size=params.test_size,
        random_state=params.random_state,
        stratify=y,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_rest,
        y_rest,
        test_size=params.valid_size,
        random_state=params.random_state,
        stratify=y_rest,
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def load_search_space(path: Path, base: ModelParams) -> list[ModelParams]:
    """Читает список кандидатов гиперпараметров (JSON-массив объектов).

    Каждый кандидат переопределяет поля базовых параметров из config.json.
    """
    overrides = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(overrides, list) or not overrides:
        raise ValueError(f"{path}: ожидается непустой JSON-массив кандидатов")
    # model_validate, а не model_copy: опечатки и невалидные значения
    # в кандидатах должны падать сразу (extra="forbid" + ограничения полей)
    return [ModelParams.model_validate({**base.model_dump(), **o}) for o in overrides]


def select_best(
    results: Sequence[dict[str, float]], metric: str = SELECTION_METRIC
) -> int:
    """Индекс кандидата с максимальной метрикой на валидации."""
    if not results:
        raise ValueError("Нет результатов для выбора лучшей модели")
    return max(range(len(results)), key=lambda i: results[i][metric])
