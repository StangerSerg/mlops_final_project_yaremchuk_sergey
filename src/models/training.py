import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src.schemas.config import ModelParams, SplitParams

EARLY_STOPPING_ROUNDS = 100


def split_to_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    params: SplitParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет выборку на обучающую и тестовую части.

    Разбиение стратифицировано по таргету: положительных примеров очень
    мало, и без стратификации их доля в тесте может заметно «гулять».

    Args:
        x: Матрица признаков.
        y: Целевая переменная.
        params: Параметры разбиения (test_size, random_state).

    Returns:
        Кортеж (x_train, x_test, y_train, y_test).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params.test_size,
        random_state=params.random_state,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: ModelParams,
    eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
) -> CatBoostClassifier:
    """
    Обучает классификатор CatBoost на переданной выборке.

    Args:
        x_train: Матрица признаков для обучения.
        y_train: Целевая переменная для обучения.
        params: Гиперпараметры модели.
        eval_set: Валидационная выборка. Если задана — включается
            early stopping и сохраняется лучшая итерация.

    Returns:
        Обученная модель ``CatBoostClassifier``.
    """
    model = CatBoostClassifier(**params.model_dump(exclude_none=True))
    if eval_set is None:
        model.fit(Pool(x_train, y_train))
    else:
        model.fit(
            Pool(x_train, y_train),
            eval_set=Pool(*eval_set),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
    return model
