import json
import logging

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.utils.logging import JsonFormatter
from tests.conftest import KNOWN_ITEM, UNKNOWN_ITEM


def _value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_prometheus_format(client: TestClient) -> None:
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text
    assert "recsys_prediction_score_bucket" in response.text


def test_request_counted_with_route_template(client: TestClient) -> None:
    labels = {"method": "GET", "handler": "/v2/models/{model_name}/ready"}
    before = _value("http_requests_total", **labels, status="200")

    client.get("/v2/models/ranker/ready")

    assert _value("http_requests_total", **labels, status="200") == before + 1
    assert _value("http_request_duration_seconds_count", **labels) >= 1


def test_prediction_updates_score_histogram(client: TestClient) -> None:
    before_count = _value("recsys_predictions_total", endpoint="predict_purchase")
    before_score = _value("recsys_prediction_score_count", endpoint="predict_purchase")
    before_hour = _value("recsys_prediction_requests_by_hour_total", hour="14")

    client.get(
        "/predict_purchase", params={"itemid": KNOWN_ITEM, "hour": 14, "weekday": 3}
    )

    assert _value("recsys_predictions_total", endpoint="predict_purchase") == (
        before_count + 1
    )
    assert _value("recsys_prediction_score_count", endpoint="predict_purchase") == (
        before_score + 1
    )
    assert _value("recsys_prediction_requests_by_hour_total", hour="14") == (
        before_hour + 1
    )


def test_unknown_item_counted_as_error(client: TestClient) -> None:
    labels = {"endpoint": "predict_purchase", "reason": "unknown_item"}
    before = _value("recsys_prediction_errors_total", **labels)

    client.get(
        "/predict_purchase", params={"itemid": UNKNOWN_ITEM, "hour": 1, "weekday": 1}
    )

    assert _value("recsys_prediction_errors_total", **labels) == before + 1


def test_model_not_loaded_counted_as_error(client_without_model: TestClient) -> None:
    labels = {"endpoint": "/predict_purchase", "reason": "model_not_loaded"}
    before = _value("recsys_prediction_errors_total", **labels)

    client_without_model.get(
        "/predict_purchase", params={"itemid": KNOWN_ITEM, "hour": 1, "weekday": 1}
    )

    assert _value("recsys_prediction_errors_total", **labels) == before + 1


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.makeLogRecord(
        {
            "name": "test",
            "levelname": "INFO",
            "msg": "prediction",
            "event": "prediction",
            "score": 0.25,
        }
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "prediction"
    assert payload["event"] == "prediction"
    assert payload["score"] == 0.25
    assert payload["level"] == "INFO"
