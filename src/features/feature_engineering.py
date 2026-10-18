import logging

import numpy as np
import pandas as pd

from src.data.preprocessing import drop_inconsistent_ctr
from src.features.utils import extract_interactions
from src.utils.logging import setup_logger

INTERACTION_COLUMNS = ["timestamp", "visitorid", "itemid"]
setup_logger()
logger = logging.getLogger(__name__)


def prepare_features(
    events: pd.DataFrame,
    item_props: pd.DataFrame,
    window_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собирает признаки для обучения модели ранжирования товаров.

    Из потока событий извлекает просмотры и покупки, строит целевую
    переменную (купил ли пользователь товар в течение window_hours
    после просмотра), добавляет агрегированную популярность товаров и
    их свойства. Возвращает два датафрейма: признаки товаров и
    обогащённый датафрейм просмотров с target.

    Args:
        events: Датафрейм событий с колонками timestamp, visitorid,
            itemid, event и is_purchase_event. Должен быть
            предварительно обработан preprocess_events.
        item_props: Датафрейм свойств товаров с колонками itemid,
            categoryid и available. Должен быть предварительно
            обработан preprocess_properties. Пропуски в categoryid
            и available заполняются значениями -1 и 0
            соответственно.
        window_hours: Размер окна в часах, в течение которого покупка
            после просмотра считается целевым событием.

    Returns:
        Кортеж из двух датафреймов:

        - item_features: признаки товаров с колонками itemid,
          views, purchases, ctr, categoryid, available.
        - views: просмотры с целевой переменной target и
          добавленными признаками товара, а также hour и weekday,
          извлечёнными из timestamp. Строки, для которых не удалось
          получить популярность товара (например, из-за фильтрации
          аномального ctr), удаляются.

    Raises:
        KeyError: Если во входных датафреймах отсутствуют обязательные
            колонки (timestamp, visitorid, itemid, event,
            is_purchase_event для events; itemid для
            item_props).

    Note:
        Для товаров, отсутствующих в item_props, значения
        categoryid и available заполняются как -1 и 0.
        Это позволяет отличать «нет данных» от реальных значений.
    """
    views = extract_interactions(
        events,
        events["event"].str.lower() == "view",
        INTERACTION_COLUMNS
    )
    purchases = extract_interactions(
        events,
        events["is_purchase_event"],
        INTERACTION_COLUMNS
    )

    views = build_target(views, purchases, window_hours=window_hours)
    pop = compute_item_popularity(events)

    item_features = pop.merge(item_props, on="itemid", how="left")
    item_features["available"] = item_features["available"].fillna(0).astype(int)
    item_features["categoryid"] = item_features["categoryid"].fillna(-1).astype(int)

    views = views.merge(item_features, on="itemid", how="left")
    views = views.dropna(subset=["views", "purchases", "ctr"]).reset_index(drop=True)
    views["available"] = views["available"].fillna(0).astype(int)
    views["categoryid"] = views["categoryid"].fillna(-1).astype(int)

    views["hour"] = views["timestamp"].dt.hour
    views["weekday"] = views["timestamp"].dt.dayofweek

    report_ctr_anomalies(views)

    return item_features, views


def build_target(
    views: pd.DataFrame, purchases: pd.DataFrame, window_hours: int = 24
) -> pd.DataFrame:
    """Создаёт целевую переменную: купил ли пользователь товар
     в течение 24 часов после просмотра.
    """
    window = pd.Timedelta(hours=window_hours)

    purchases = purchases.rename(columns={"timestamp": "purchase_time"})
    df = views.merge(purchases, on=["visitorid", "itemid"], how="left")

    df["target"] = (
        (df["purchase_time"].notna())
        & (df["purchase_time"] > df["timestamp"])
        & (df["purchase_time"] <= df["timestamp"] + window)
    ).astype(int)

    df = df.drop(columns=["purchase_time"])
    return df


def compute_item_popularity(events: pd.DataFrame) -> pd.DataFrame:
    """Считает простую популярность товаров."""
    views = (
        events[events["event"].str.lower() == "view"]
        .groupby("itemid")
        .size()
        .rename("views")
    )
    buys = (
        events[events["is_purchase_event"]].groupby("itemid").size().rename("purchases")
    )

    df = pd.concat([views, buys], axis=1).fillna(0)
    df["ctr"] = df["purchases"] / np.maximum(df["views"], 1)
    df = df.reset_index()
    df = drop_inconsistent_ctr(df, logger, "ctr")

    return df


def report_ctr_anomalies(views: pd.DataFrame) -> pd.DataFrame:
    """Выводит в консоль все строки с ctr > 1
     и логирует статистику по уникальным itemid.
     """
    anomalies = (
        views[views["ctr"] > 1]
        .sort_values("ctr", ascending=False)
        .reset_index(drop=True)
    )

    if anomalies.empty:
        logger.info("Аномалий ctr > 1 не найдено")
        return anomalies

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    logger.info(f"\n=== Строки с ctr > 1: {len(anomalies)} ===")
    logger.info(anomalies.to_string(index=False))

    unique_items = anomalies["itemid"].nunique()
    logger.warning(
        "ctr > 1: строк=%d, уникальных itemid=%d, макс ctr=%.4f",
        len(anomalies),
        unique_items,
        anomalies["ctr"].max(),
    )

    logger.info(f"\nУникальных itemid с ctr > 1: {unique_items}")

    return anomalies