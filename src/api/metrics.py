"""Prometheus-метрики сервиса.

Технические: RPS, latency, ошибки (по ручке и статусу), запросы в работе.
Модельные: распределение score, частота неизвестных товаров,
распределение входов (час, день недели) — для отслеживания дрейфа
поведения пользователей и модели.
"""

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from src.version import APP_VERSION, SERVICE_NAME

# ---------- технические ----------
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP-запросы по ручке, методу и статусу",
    ["method", "handler", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Время обработки HTTP-запроса",
    ["method", "handler"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
HTTP_IN_PROGRESS = Gauge(
    "http_requests_in_progress", "HTTP-запросы в обработке", ["method"]
)

# ---------- модель ----------
APP_INFO = Info("recsys_app", "Версия сервиса")
APP_INFO.info({"service": SERVICE_NAME, "version": APP_VERSION})
MODEL_LOADED = Gauge("recsys_model_loaded", "1 — модель и признаки загружены")
FEATURE_STORE_ITEMS = Gauge("recsys_feature_store_items", "Товаров в feature store")
PREDICTIONS = Counter("recsys_predictions_total", "Выданные предсказания", ["endpoint"])
PREDICTION_SCORE = Histogram(
    "recsys_prediction_score",
    "Распределение предсказанной вероятности покупки",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
)
PREDICTION_ERRORS = Counter(
    "recsys_prediction_errors_total",
    "Ошибки инференса по причине (unknown_item, model_not_loaded)",
    ["endpoint", "reason"],
)
REQUESTS_BY_HOUR = Counter(
    "recsys_prediction_requests_by_hour_total",
    "Запросы предсказаний по часу (дрейф поведения пользователей)",
    ["hour"],
)
REQUESTS_BY_WEEKDAY = Counter(
    "recsys_prediction_requests_by_weekday_total",
    "Запросы предсказаний по дню недели",
    ["weekday"],
)


def observe_predictions(
    endpoint: str, hours: list[int], weekdays: list[int], scores: list[float]
) -> None:
    PREDICTIONS.labels(endpoint).inc(len(scores))
    for hour, weekday, score in zip(hours, weekdays, scores, strict=True):
        PREDICTION_SCORE.labels(endpoint).observe(score)
        REQUESTS_BY_HOUR.labels(str(hour)).inc()
        REQUESTS_BY_WEEKDAY.labels(str(weekday)).inc()


def observe_error(endpoint: str, reason: str, count: int = 1) -> None:
    PREDICTION_ERRORS.labels(endpoint, reason).inc(count)


def set_model_state(loaded: bool, items: int = 0) -> None:
    MODEL_LOADED.set(1 if loaded else 0)
    FEATURE_STORE_ITEMS.set(items)


def _handler_name(request: Request) -> str:
    # шаблон пути (/v2/models/{model_name}/infer), а не сырой URL —
    # чтобы не раздувать кардинальность меток
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def setup_metrics(app: FastAPI) -> None:
    """Middleware со счётчиками запросов и ручка /metrics."""

    @app.middleware("http")
    async def prometheus_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        method = request.method
        HTTP_IN_PROGRESS.labels(method).inc()
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            handler = _handler_name(request)
            HTTP_LATENCY.labels(method, handler).observe(time.perf_counter() - start)
            HTTP_REQUESTS.labels(method, handler, status).inc()
            HTTP_IN_PROGRESS.labels(method).dec()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
