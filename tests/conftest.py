from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_optional_artifacts
from src.main import app
from src.models.artifacts import ServiceArtifacts

FEATURES = ["views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"]
KNOWN_ITEM = 356475
OTHER_ITEM = 98113
UNKNOWN_ITEM = 1


class FakeModel:
    """Модель-заглушка: всегда возвращает фиксированную вероятность."""

    def __init__(self, proba: float = 0.25) -> None:
        self.proba = proba
        self.last_x: pd.DataFrame | None = None

    def predict_proba(self, x: Any) -> np.ndarray:
        self.last_x = x
        return np.array([[1 - self.proba, self.proba]] * len(x))


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def artifacts(fake_model: FakeModel) -> ServiceArtifacts:
    item_features = pd.DataFrame(
        {
            "itemid": [KNOWN_ITEM, OTHER_ITEM],
            "views": [10.0, 5.0],
            "purchases": [1.0, 0.0],
            "ctr": [0.1, 0.0],
            "categoryid": [1, 2],
            "available": [1, 0],
        }
    ).set_index("itemid")
    return ServiceArtifacts(
        model=fake_model, item_features=item_features, features=FEATURES
    )


@pytest.fixture
def client(artifacts: ServiceArtifacts) -> Iterator[TestClient]:
    # TestClient без контекстного менеджера не запускает lifespan,
    # поэтому реальная модель из S3 не загружается.
    app.dependency_overrides[get_optional_artifacts] = lambda: artifacts
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_model() -> Iterator[TestClient]:
    app.dependency_overrides[get_optional_artifacts] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()
