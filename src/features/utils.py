import pandas as pd


def extract_interactions(
    events: pd.DataFrame,
    mask: pd.Series,
    interaction_columns: list[str]
) -> pd.DataFrame:
    """Выбирает из событий нужные колонки по булевой маске.

    Args:
        events: Исходный DataFrame с событиями.
        mask: Булева маска отбора строк (той же длины, что и events).

    Returns:
        Новый DataFrame с колонками timestamp, visitorid, itemid.
    """
    return events.loc[mask, interaction_columns].copy()