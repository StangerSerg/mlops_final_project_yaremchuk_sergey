import pandas as pd

from src.schemas.data import training_schema


def validate_data(
        df: pd.DataFrame,
        required_columns: list[str]
) -> None:
    missed_columns = [rc for rc in required_columns if rc not in df]
    if missed_columns:
        raise ValueError(
            f"Required columns are missed: {missed_columns}"
        )

    training_schema.validate(df)

