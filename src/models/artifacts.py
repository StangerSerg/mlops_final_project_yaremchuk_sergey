from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from src.data.ingestion import load_parquet_data
from src.schemas.config import Config
from src.utils.storage import ArtifactStorage


class ProbabilisticClassifier(Protocol):
    def predict_proba(self, x: Any) -> np.ndarray: ...


class UnknownItemsError(KeyError):
    """Товаров нет в feature store."""

    def __init__(self, itemids: list[int]) -> None:
        super().__init__(itemids)
        self.itemids = itemids


@dataclass(frozen=True)
class ServiceArtifacts:
    """Всё, что нужно сервису для инференса.

    Attributes:
        model: Модель с методом predict_proba.
        item_features: Признаки товаров, индекс — itemid.
        features: Порядок признаков, на котором обучалась модель.
    """

    model: ProbabilisticClassifier
    item_features: pd.DataFrame
    features: list[str]

    def predict_batch(
        self,
        itemids: Sequence[int],
        hours: Sequence[int],
        weekdays: Sequence[int],
    ) -> list[float]:
        """Вероятности покупки для пачки (itemid, hour, weekday).

        Raises:
            UnknownItemsError: часть товаров отсутствует в feature store.
        """
        missing = [i for i in itemids if i not in self.item_features.index]
        if missing:
            raise UnknownItemsError(missing)
        rows = self.item_features.loc[list(itemids)].copy()
        rows["hour"] = list(hours)
        rows["weekday"] = list(weekdays)
        x = rows[self.features].astype(float)
        return [float(p) for p in self.model.predict_proba(x)[:, 1]]

    def predict_purchase(self, itemid: int, hour: int, weekday: int) -> float | None:
        """Вероятность покупки товара; None, если товара нет в feature store."""
        try:
            return self.predict_batch([itemid], [hour], [weekday])[0]
        except UnknownItemsError:
            return None


def load_service_artifacts(storage: ArtifactStorage, config: Config) -> ServiceArtifacts:
    """Загружает модель и признаки товаров из S3 (или локального fallback)."""
    model = joblib.load(storage.fetch("models", config.model_file))
    item_features = load_parquet_data(
        storage.fetch("processed", config.item_features_file)
    ).set_index("itemid")
    return ServiceArtifacts(
        model=model, item_features=item_features, features=config.features
    )
