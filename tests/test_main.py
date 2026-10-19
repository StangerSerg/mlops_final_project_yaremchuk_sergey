from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.models.artifacts import ServiceArtifacts
from tests.conftest import FEATURES, KNOWN_ITEM, UNKNOWN_ITEM, FakeModel

# ---------- /predict_purchase ----------


def test_predict_purchase_returns_200(client: TestClient) -> None:
    response = client.get(
        "/predict_purchase", params={"itemid": KNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert response.status_code == 200


def test_predict_purchase_returns_probability(client: TestClient) -> None:
    response = client.get(
        "/predict_purchase", params={"itemid": KNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert response.json() == {"itemid": KNOWN_ITEM, "purchase_probability": 0.25}


def test_predict_purchase_passes_features_in_training_order(
    client: TestClient, fake_model: FakeModel
) -> None:
    client.get(
        "/predict_purchase", params={"itemid": KNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert fake_model.last_x is not None
    assert list(fake_model.last_x.columns) == FEATURES
    row = fake_model.last_x.iloc[0]
    assert (row["hour"], row["weekday"]) == (14.0, 3.0)


def test_predict_purchase_unknown_item_returns_404(client: TestClient) -> None:
    response = client.get(
        "/predict_purchase", params={"itemid": UNKNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "item not found"}


@pytest.mark.parametrize(
    "params",
    [
        {"itemid": KNOWN_ITEM, "hour": 24, "weekday": 3},
        {"itemid": KNOWN_ITEM, "hour": -1, "weekday": 3},
        {"itemid": KNOWN_ITEM, "hour": 14, "weekday": 7},
        {"itemid": -5, "hour": 14, "weekday": 3},
        {"itemid": "abc", "hour": 14, "weekday": 3},
    ],
)
def test_predict_purchase_invalid_params_return_422(
    client: TestClient, params: dict[str, Any]
) -> None:
    response = client.get("/predict_purchase", params=params)

    assert response.status_code == 422


def test_predict_purchase_without_model_returns_503(
    client_without_model: TestClient,
) -> None:
    response = client_without_model.get(
        "/predict_purchase", params={"itemid": KNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert response.status_code == 503


# ---------- /health ----------


def test_health_returns_200_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ranker_loaded": True,
        "items_in_feature_store": 2,
    }


def test_health_returns_503_when_model_not_loaded(
    client_without_model: TestClient,
) -> None:
    response = client_without_model.get("/health")

    assert response.status_code == 503
    assert response.json()["ranker_loaded"] is False


# ---------- /version ----------


def test_version_returns_service_and_version(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert set(response.json()) == {"service", "version"}


# ---------- ServiceArtifacts ----------


def test_artifacts_predict_unknown_item_returns_none(
    artifacts: ServiceArtifacts,
) -> None:
    assert artifacts.predict_purchase(UNKNOWN_ITEM, 12, 3) is None
