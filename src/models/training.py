from typing import TypedDict

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src.schemas.output_schemas import ModelParams, SplitParams


class TrainParams(TypedDict):
    depth: int
    learning_rate: float
    loss_function: str
    verbose: bool


def split_to_train_test(
        x: pd.DataFrame,
        y: pd.Series,
        params: SplitParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
        Разделяет выборку на обучающую и тестовую (или валидационную) части.
        Обёртка над sklearn.model_selection.train_test_split,
        которая берёт параметры разбиения из словаря конфигурации.

        Args:
            x: Матрица признаков.
            y: Целевая переменная.
            params: Словарь с параметрами разбиения. Ожидаемые ключи:
                - test_size: float: доля тестовой части.
                - random_state: int: seed для воспроизводимости.

        Returns:
            Кортеж из четырёх элементов в порядке, который возвращает
            train_test_split:
            (x_train, x_test, y_train, y_test).
        """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params.test_size,
        random_state=params.random_state
    )
    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: ModelParams
) -> CatBoostClassifier:
    """
    Обучает классификатор CatBoost на переданной выборке.

    Создаёт CatBoostClassifier с гиперпараметрами из params,
    упаковывает данные в Pool (нативный формат CatBoost) и запускает обучение.

    Args:
        x_train: Матрица признаков для обучения.
        y_train: Целевая переменная для обучения.
        params: Словарь с гиперпараметрами модели. Ожидаемые ключи:
            - depth: int: глубина деревьев.
            - learning_rate: float: скорость обучения.
            - loss_function: str: функция потерь, например Logloss.
            - verbose: bool: частота логирования обучения.

    Returns:
        Обученная модель ``CatBoostClassifier``.
    """
    model: CatBoostClassifier = CatBoostClassifier(
        depth=params.depth,
        learning_rate=params.learning_rate,
        loss_function=params.loss_function,
        verbose=params.verbose
    )
    train_pool = Pool(x_train, y_train)
    model.fit(train_pool)

    return model