import logging
from collections.abc import Sequence

from src.api.metrics import observe_error, observe_predictions
from src.models.artifacts import ServiceArtifacts, UnknownItemsError

logger = logging.getLogger(__name__)


def predict(
    artifacts: ServiceArtifacts,
    itemids: Sequence[int],
    hours: Sequence[int],
    weekdays: Sequence[int],
    endpoint: str,
) -> list[float]:
    """Единая точка инференса для всех эндпоинтов: метрики и лог предсказаний.

    Raises:
        UnknownItemsError: часть товаров отсутствует в feature store.
    """
    try:
        probs = artifacts.predict_batch(itemids, hours, weekdays)
    except UnknownItemsError as e:
        observe_error(endpoint, "unknown_item", len(e.itemids))
        logger.warning(
            "unknown items",
            extra={
                "event": "prediction_error",
                "endpoint": endpoint,
                "reason": "unknown_item",
                "itemids": e.itemids[:20],
            },
        )
        raise

    observe_predictions(endpoint, list(hours), list(weekdays), probs)
    for itemid, hour, weekday, prob in zip(
        itemids, hours, weekdays, probs, strict=True
    ):
        logger.info(
            "prediction",
            extra={
                "event": "prediction",
                "endpoint": endpoint,
                "itemid": itemid,
                "hour": hour,
                "weekday": weekday,
                "score": round(prob, 6),
            },
        )
    return probs
