import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api import v2
from src.api.deps import get_artifacts, get_optional_artifacts
from src.api.inference import predict
from src.api.metrics import set_model_state, setup_metrics
from src.models.artifacts import (
    ServiceArtifacts,
    UnknownItemsError,
    load_service_artifacts,
)
from src.schemas.api import (
    ErrorResponse,
    HealthResponse,
    PredictPurchaseRequest,
    PredictPurchaseResponse,
    VersionResponse,
)
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.settings import get_settings
from src.utils.storage import ArtifactStorage
from src.version import APP_VERSION, SERVICE_NAME

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загружает модель и признаки товаров при старте сервиса.

    Если артефакты загрузить не удалось, сервис всё равно стартует,
    но /health, /v2/health/ready и инференс отвечают 503.
    """
    settings = get_settings()
    setup_logger(settings.log_path, settings.log_format)
    try:
        storage = ArtifactStorage(settings)
        app.state.artifacts = load_service_artifacts(
            storage, load_config(settings.config_path)
        )
        items = len(app.state.artifacts.item_features)
        set_model_state(loaded=True, items=items)
        logger.info("Артефакты загружены, товаров: %d", items)
    except Exception:
        logger.exception("Не удалось загрузить артефакты модели")
        app.state.artifacts = None
        set_model_state(loaded=False)
    yield


app = FastAPI(
    title="Retail Recommendation API",
    version=APP_VERSION,
    description="API для оценки вероятности покупки товара.",
    lifespan=lifespan,
)
app.include_router(v2.router)
app.add_exception_handler(v2.V2Error, v2.v2_error_handler)
setup_metrics(app)


@app.get(
    "/health",
    tags=["system"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
def health_check(
    artifacts: Annotated[ServiceArtifacts | None, Depends(get_optional_artifacts)],
) -> HealthResponse | JSONResponse:
    """Проверка состояния сервиса и доступности модели."""
    if artifacts is None:
        body = HealthResponse(
            status="unavailable", ranker_loaded=False, items_in_feature_store=0
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    return HealthResponse(
        status="ok",
        ranker_loaded=True,
        items_in_feature_store=len(artifacts.item_features),
    )


@app.get("/version", tags=["system"], response_model=VersionResponse)
def version() -> VersionResponse:
    """Возвращает текущую версию сервиса."""
    return VersionResponse(service=SERVICE_NAME, version=APP_VERSION)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "Retail recommender API running", "version": APP_VERSION}


@app.get(
    "/predict_purchase",
    tags=["model"],
    response_model=PredictPurchaseResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_purchase(
    request: Annotated[PredictPurchaseRequest, Depends()],
    artifacts: Annotated[ServiceArtifacts, Depends(get_artifacts)],
) -> PredictPurchaseResponse:
    """Вероятность покупки товара в заданный час и день недели.

    Args:
        request: параметры запроса (itemid, hour, weekday).
    """
    try:
        [prob] = predict(
            artifacts,
            [request.itemid],
            [request.hour],
            [request.weekday],
            endpoint="predict_purchase",
        )
    except UnknownItemsError as e:
        raise HTTPException(status_code=404, detail="item not found") from e
    return PredictPurchaseResponse(itemid=request.itemid, purchase_probability=prob)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
