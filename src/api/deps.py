from typing import Annotated

from fastapi import Depends, HTTPException, Request

from src.api.metrics import observe_error
from src.models.artifacts import ServiceArtifacts


def get_optional_artifacts(request: Request) -> ServiceArtifacts | None:
    """Артефакты модели, загруженные при старте (None — не загрузились)."""
    return getattr(request.app.state, "artifacts", None)


def endpoint_name(request: Request) -> str:
    return getattr(request.scope.get("route"), "path", request.url.path)


def get_artifacts(
    request: Request,
    artifacts: Annotated[ServiceArtifacts | None, Depends(get_optional_artifacts)],
) -> ServiceArtifacts:
    if artifacts is None:
        observe_error(endpoint_name(request), "model_not_loaded")
        raise HTTPException(status_code=503, detail="model is not loaded")
    return artifacts
