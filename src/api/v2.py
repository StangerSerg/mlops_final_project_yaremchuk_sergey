"""Open Inference Protocol (KServe V2) поверх той же модели.

Нужен для деплоя в Kubernetes: health/ready-пробы и единый формат
инференса ``POST /v2/models/<MODEL_NAME>/infer``. Для ручной проверки
есть упрощённый ``GET /v2/models/<MODEL_NAME>/infer?itemid=..&hour=..&weekday=..``.
"""

import math
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.deps import endpoint_name, get_optional_artifacts
from src.api.inference import predict
from src.api.metrics import observe_error
from src.models.artifacts import ServiceArtifacts, UnknownItemsError
from src.schemas.api import PredictPurchaseRequest
from src.schemas.v2 import (
    InferOutputTensor,
    InferRequest,
    InferResponse,
    LiveResponse,
    ModelMetadataResponse,
    ReadyResponse,
    ServerMetadataResponse,
    TensorMetadata,
    V2ErrorResponse,
)
from src.utils.settings import get_settings
from src.version import APP_VERSION, SERVICE_NAME

# Входы модели: имя -> допустимый диапазон значений
INPUT_RANGES: dict[str, tuple[int, int | None]] = {
    "itemid": (0, None),
    "hour": (0, 23),
    "weekday": (0, 6),
}
OUTPUT_NAME = "purchase_probability"

ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": V2ErrorResponse},
    404: {"model": V2ErrorResponse},
    503: {"model": V2ErrorResponse},
}


class V2Error(Exception):
    """Ошибка V2-протокола: ответ ``{"error": "<message>"}``."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


async def v2_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, V2Error)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


def check_model_name(model_name: str) -> str:
    if model_name != get_settings().model_name:
        raise V2Error(404, f"model '{model_name}' not found")
    return model_name


def require_artifacts(
    request: Request,
    artifacts: Annotated[ServiceArtifacts | None, Depends(get_optional_artifacts)],
) -> ServiceArtifacts:
    if artifacts is None:
        observe_error(endpoint_name(request), "model_not_loaded")
        raise V2Error(503, "model is not loaded")
    return artifacts


router = APIRouter(prefix="/v2", tags=["v2 inference protocol"])


@router.get("", response_model=ServerMetadataResponse)
def server_metadata() -> ServerMetadataResponse:
    return ServerMetadataResponse(name=SERVICE_NAME, version=APP_VERSION)


@router.get("/health/live", response_model=LiveResponse)
def server_live() -> LiveResponse:
    """Liveness: процесс сервиса отвечает."""
    return LiveResponse(live=True)


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
)
def server_ready(
    artifacts: Annotated[ServiceArtifacts | None, Depends(get_optional_artifacts)],
) -> ReadyResponse | JSONResponse:
    """Readiness: модель загружена и готова к инференсу."""
    if artifacts is None:
        return JSONResponse(status_code=503, content={"ready": False})
    return ReadyResponse(ready=True)


@router.get(
    "/models/{model_name}", response_model=ModelMetadataResponse, responses=ERRORS
)
def model_metadata(
    model_name: Annotated[str, Depends(check_model_name)],
) -> ModelMetadataResponse:
    return ModelMetadataResponse(
        name=model_name,
        versions=[get_settings().model_version],
        platform="catboost",
        inputs=[
            TensorMetadata(name=name, datatype="INT64", shape=[-1])
            for name in INPUT_RANGES
        ],
        outputs=[TensorMetadata(name=OUTPUT_NAME, datatype="FP64", shape=[-1])],
    )


@router.get(
    "/models/{model_name}/ready", response_model=ReadyResponse, responses=ERRORS
)
def model_ready(
    model_name: Annotated[str, Depends(check_model_name)],
    artifacts: Annotated[ServiceArtifacts, Depends(require_artifacts)],
) -> ReadyResponse:
    return ReadyResponse(ready=True)


@router.post(
    "/models/{model_name}/infer", response_model=InferResponse, responses=ERRORS
)
def infer(
    body: InferRequest,
    model_name: Annotated[str, Depends(check_model_name)],
    artifacts: Annotated[ServiceArtifacts, Depends(require_artifacts)],
) -> InferResponse:
    """Инференс по V2: входы itemid, hour, weekday (INT64, shape [N])."""
    columns = parse_inputs(body)
    return _infer(artifacts, model_name, columns, request_id=body.id)


@router.get(
    "/models/{model_name}/infer", response_model=InferResponse, responses=ERRORS
)
def infer_get(
    request: Annotated[PredictPurchaseRequest, Depends()],
    model_name: Annotated[str, Depends(check_model_name)],
    artifacts: Annotated[ServiceArtifacts, Depends(require_artifacts)],
) -> InferResponse:
    """Упрощённый инференс одной записи через query-параметры (для проверки)."""
    columns = {
        "itemid": [request.itemid],
        "hour": [request.hour],
        "weekday": [request.weekday],
    }
    return _infer(artifacts, model_name, columns, request_id=None)


def _infer(
    artifacts: ServiceArtifacts,
    model_name: str,
    columns: dict[str, list[int]],
    request_id: str | None,
) -> InferResponse:
    try:
        probs = predict(
            artifacts,
            columns["itemid"],
            columns["hour"],
            columns["weekday"],
            endpoint="v2_infer",
        )
    except UnknownItemsError as e:
        raise V2Error(404, f"items not found: {e.itemids[:20]}") from e
    return InferResponse(
        model_name=model_name,
        model_version=get_settings().model_version,
        id=request_id,
        outputs=[
            InferOutputTensor(
                name=OUTPUT_NAME, shape=[len(probs)], datatype="FP64", data=probs
            )
        ],
    )


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def parse_inputs(body: InferRequest) -> dict[str, list[int]]:
    """Проверяет V2-входы и возвращает колонки itemid/hour/weekday.

    Raises:
        V2Error(400): нет нужного входа, неверный тип/shape/диапазон,
            разная длина входов.
    """
    by_name = {tensor.name: tensor for tensor in body.inputs}
    missing = [name for name in INPUT_RANGES if name not in by_name]
    if missing:
        raise V2Error(400, f"missing inputs: {missing}")

    columns: dict[str, list[int]] = {}
    for name, (low, high) in INPUT_RANGES.items():
        tensor = by_name[name]
        if not all(_is_integer(v) for v in tensor.data):
            raise V2Error(400, f"input '{name}': integer values expected")
        values = [int(v) for v in tensor.data]
        if math.prod(tensor.shape) != len(values):
            raise V2Error(
                400, f"input '{name}': shape {tensor.shape} does not match data"
            )
        if any(v < low or (high is not None and v > high) for v in values):
            bounds = f"[{low}, {high}]" if high is not None else f">= {low}"
            raise V2Error(400, f"input '{name}': values must be {bounds}")
        columns[name] = values

    if len({len(v) for v in columns.values()}) != 1:
        raise V2Error(400, "inputs must have the same length")
    if not columns["itemid"]:
        raise V2Error(400, "inputs must not be empty")
    return columns
