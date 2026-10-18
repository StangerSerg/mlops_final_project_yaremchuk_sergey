from typing import Any

import pandas as pd

INTERACTION_COLUMNS = ["itemid", "property", "value"]
SIMPLE_PROPERTIES = ["itemid", "property"]


def preprocess_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет записи о событиях, не относящихся к покупкам товаров.

    Args:
        df: Исходный DataFrame из данных по событиям.

    Returns:
        Очищенный DataFrame без лишних событий.
    """
    df.columns = [c.lower().strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["is_purchase_event"] = (
        df["event"].astype(str).str.lower().isin(["purchase", "transaction"])
        | df.get("transactionid", pd.Series([None] * len(df))).notna()
    )

    return df


def preprocess_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка таблицы свойств (оставляем только простые поля).

    Args:
        df: Исходный DataFrame из данных по свойствам.

    Returns:
        Очищенный DataFrame.
    """
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[INTERACTION_COLUMNS]

    # Возьмём только базовые свойства, чтобы не взорваться по памяти
    simple_props = df[df["property"].isin(["categoryid", "available"])]
    # Последние значения свойств
    simple_props = simple_props.sort_values(SIMPLE_PROPERTIES)
    latest = simple_props.groupby(SIMPLE_PROPERTIES, as_index=False).last()

    # Поворот: property -> колонка
    pivot = latest.pivot_table(
        index="itemid", columns="property", values="value", aggfunc="last"
    )
    pivot = pivot.reset_index()

    # Убедимся, что названия колонок корректные
    pivot.columns.name = None

    # Приведение типов
    if "categoryid" in pivot.columns:
        pivot["categoryid"] = pd.to_numeric(
            pivot["categoryid"],
            errors="coerce"
        ).fillna(-1).astype(int)
    else:
        pivot["categoryid"] = -1

    if "available" in pivot.columns:
        pivot["available"] = pd.to_numeric(
            pivot["available"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        pivot["available"] = 0

    return pivot


def drop_inconsistent_ctr(
    df: pd.DataFrame,
    logger: Any,
    ctr_column: str = "ctr",
    max_ctr: float = 1.0,
) -> pd.DataFrame:
    """Удаляет строки, где ctr превышает разумный максимум.

    ctr > 1 означает, что покупок больше, чем просмотров, — это признак
    неполных view-логов, а не валидная популярность.

    Args:
        df: DataFrame с колонкой ctr.
        logger: логгер из внешней функции чтобы не дублировать логику его инциализации.
        ctr_column: имя колонки с ctr.
        max_ctr: верхняя допустимая граница.

    Returns:
        DataFrame без аномальных строк.
    """
    mask = df[ctr_column] > max_ctr
    if mask.any():
        bad_items = df.loc[
            mask, "itemid"
        ].unique().tolist() if "itemid" in df.columns else []
        logger.info(
            "drop_inconsistent_ctr: отброшено %d "
            "строк (уникальных itemid: %d) с %s > %.2f: %s",
            int(mask.sum()),
            len(bad_items),
            ctr_column,
            max_ctr,
            bad_items[:20],  # ограничим вывод, если товаров вдруг много
        )
    return df.loc[~mask].reset_index(drop=True)