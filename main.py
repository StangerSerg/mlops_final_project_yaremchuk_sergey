import logging
from typing import Annotated

from fastapi import Depends, FastAPI

from src.models.loaded_models import item_features, ranker

# Возможно надо еще импортировать prometheus, pydantic (схемы)
from src.schemas.output_schemas import (
    ErrorResponse,
    PredictPurchaseRequest,
    PredictPurchaseResponse,
)

# from prometheus_client import
# Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.0.1"

app = FastAPI(
    title="Retail Recommendation API",
    version=APP_VERSION,
    description="API для выдачи рекомендаций и вероятностей покупки.",
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str | dict[str, int]]:
    """
    Проверка состояния сервиса и доступности моделей.
    """
    from src.models.loaded_models import item_features, ranker

    ok = {
        "ranker_loaded": ranker is not None,
        "items_in_feature_store": len(item_features),
    }
    logger.info(f"Health check: {ok}")
    return {"status": "ok", "details": ok}


@app.get("/version", tags=["system"])
def version() -> dict[str, str]:
    """
    Возвращает текущую версию сервиса.
    """
    return {"version": APP_VERSION, "service": "retail_recsys"}


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "Retail recommender API running", "version": APP_VERSION}


@app.get("/predict_purchase", response_model=PredictPurchaseResponse)
def predict_purchase(
    request: Annotated[PredictPurchaseRequest, Depends()],
) -> PredictPurchaseResponse | ErrorResponse:
    """Предсказываем вероятность покупки товара.

    Note:
        Предсказание вероятности покупки товара в конкретный день недели и время.

    Args:
        request: параметры запроса (itemid, hour, weekday).
    """
    row = item_features[item_features["itemid"] == request.itemid].copy()
    if row.empty:
        return ErrorResponse(error="item not found")

    row["hour"] = request.hour
    row["weekday"] = request.weekday
    x = row[
        ["views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"]
    ].astype(float)
    prob = float(ranker.predict_proba(x)[0][1])

    return PredictPurchaseResponse(
        itemid=request.itemid,
        purchase_probability=prob,
    )


@app.get("/metrics", tags=["system"])
def metrics() -> None:
    """Надо как-то все логировать"""
    # Заполни меня - основная логика по которой мы отдаем метрики в prometheus.
    ...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=32000, workers=1)
