from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import KNOWN_ITEM, OTHER_ITEM, UNKNOWN_ITEM, FakeModel

MODEL = "ranker"
INFER_URL = f"/v2/models/{MODEL}/infer"


def _body(itemids: list[Any], hours: list[Any], weekdays: list[Any]) -> dict:
    def tensor(name: str, data: list[Any]) -> dict[str, Any]:
        return {"name": name, "shape": [len(data)], "datatype": "INT64", "data": data}

    return {
        "id": "req-1",
        "inputs": [
            tensor("itemid", itemids),
            tensor("hour", hours),
            tensor("weekday", weekdays),
        ],
    }


# ---------- health / metadata ----------


def test_live_returns_200_even_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.get("/v2/health/live")

    assert response.status_code == 200
    assert response.json() == {"live": True}


def test_ready_returns_200_when_model_loaded(client: TestClient) -> None:
    response = client.get("/v2/health/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_ready_returns_503_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.get("/v2/health/ready")

    assert response.status_code == 503
    assert response.json() == {"ready": False}


def test_model_ready_returns_200(client: TestClient) -> None:
    assert client.get(f"/v2/models/{MODEL}/ready").status_code == 200


def test_model_ready_without_model_returns_503(
    client_without_model: TestClient,
) -> None:
    response = client_without_model.get(f"/v2/models/{MODEL}/ready")

    assert response.status_code == 503
    assert "error" in response.json()


def test_server_metadata(client: TestClient) -> None:
    response = client.get("/v2")

    assert response.status_code == 200
    assert set(response.json()) >= {"name", "version"}


def test_model_metadata_describes_inputs_and_outputs(client: TestClient) -> None:
    response = client.get(f"/v2/models/{MODEL}")

    body = response.json()
    assert response.status_code == 200
    assert [i["name"] for i in body["inputs"]] == ["itemid", "hour", "weekday"]
    assert [o["name"] for o in body["outputs"]] == ["purchase_probability"]


def test_unknown_model_returns_404(client: TestClient) -> None:
    response = client.get("/v2/models/other/ready")

    assert response.status_code == 404
    assert "error" in response.json()


# ---------- POST infer ----------


def test_infer_returns_probabilities(client: TestClient) -> None:
    response = client.post(
        INFER_URL, json=_body([KNOWN_ITEM, OTHER_ITEM], [14, 3], [3, 6])
    )

    body = response.json()
    assert response.status_code == 200
    assert body["model_name"] == MODEL
    assert body["id"] == "req-1"
    [output] = body["outputs"]
    assert output["name"] == "purchase_probability"
    assert output["shape"] == [2]
    assert output["data"] == [0.25, 0.25]


def test_infer_passes_rows_to_model_in_order(
    client: TestClient, fake_model: FakeModel
) -> None:
    client.post(INFER_URL, json=_body([OTHER_ITEM, KNOWN_ITEM], [1, 2], [5, 6]))

    assert fake_model.last_x is not None
    assert fake_model.last_x["hour"].tolist() == [1.0, 2.0]
    assert fake_model.last_x["weekday"].tolist() == [5.0, 6.0]
    assert fake_model.last_x["views"].tolist() == [5.0, 10.0]


def test_infer_unknown_item_returns_404(client: TestClient) -> None:
    response = client.post(INFER_URL, json=_body([UNKNOWN_ITEM], [1], [1]))

    assert response.status_code == 404
    assert str(UNKNOWN_ITEM) in response.json()["error"]


@pytest.mark.parametrize(
    "body",
    [
        _body([KNOWN_ITEM], [24], [1]),  # hour вне диапазона
        _body([KNOWN_ITEM], [1], [7]),  # weekday вне диапазона
        _body([KNOWN_ITEM], [1.5], [1]),  # не целое
        _body([KNOWN_ITEM], ["a"], [1]),  # не число
        _body([KNOWN_ITEM, OTHER_ITEM], [1], [1]),  # разная длина
        _body([], [], []),  # пусто
        {
            "inputs": [
                {"name": "itemid", "shape": [1], "datatype": "INT64", "data": [1]}
            ]
        },
        {
            "inputs": [
                {"name": "itemid", "shape": [2], "datatype": "INT64", "data": [1]},
                {"name": "hour", "shape": [1], "datatype": "INT64", "data": [1]},
                {"name": "weekday", "shape": [1], "datatype": "INT64", "data": [1]},
            ]
        },  # shape не совпадает с data
    ],
)
def test_infer_invalid_inputs_return_400(client: TestClient, body: dict) -> None:
    response = client.post(INFER_URL, json=body)

    assert response.status_code == 400
    assert "error" in response.json()


def test_infer_without_model_returns_503(client_without_model: TestClient) -> None:
    response = client_without_model.post(INFER_URL, json=_body([KNOWN_ITEM], [1], [1]))

    assert response.status_code == 503


def test_infer_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/v2/models/other/infer", json=_body([KNOWN_ITEM], [1], [1]))

    assert response.status_code == 404


# ---------- GET infer (упрощённый) ----------


def test_infer_get_returns_200(client: TestClient) -> None:
    response = client.get(
        INFER_URL, params={"itemid": KNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert response.status_code == 200
    assert response.json()["outputs"][0]["data"] == [0.25]


def test_infer_get_unknown_item_returns_404(client: TestClient) -> None:
    response = client.get(
        INFER_URL, params={"itemid": UNKNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert response.status_code == 404
