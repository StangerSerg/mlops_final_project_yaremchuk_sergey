from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.data.validation as validation_module
from src.data.validation import validate_data


def test_validate_data_raises_when_required_column_missing() -> None:
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        validate_data(df, ["a", "b"])


def test_validate_data_error_message_contains_missing_column() -> None:
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="b"):
        validate_data(df, ["a", "b"])


def test_validate_data_lists_all_missing_columns() -> None:
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="b.*c|c.*b"):
        validate_data(df, ["a", "b", "c"])


def test_validate_data_raises_on_empty_required_columns_with_empty_df() -> None:
    df = pd.DataFrame()

    # пустой список required_columns -> пропусков нет, идём в схему
    # проверяем, что не падает на шаге проверки колонок
    with patch.object(validation_module.training_schema, "validate", MagicMock()):
        validate_data(df, [])


def test_validate_data_calls_schema_validate() -> None:
    df = pd.DataFrame({"a": [1]})

    with patch.object(validation_module.training_schema, "validate") as mocked:
        validate_data(df, ["a"])

    assert mocked.called


def test_validate_data_passes_df_to_schema() -> None:
    df = pd.DataFrame({"a": [1]})

    with patch.object(validation_module.training_schema, "validate") as mocked:
        validate_data(df, ["a"])

    args, _ = mocked.call_args
    assert args[0] is df


def test_validate_data_does_not_call_schema_when_column_missing() -> None:
    df = pd.DataFrame({"a": [1]})

    with patch.object(validation_module.training_schema, "validate") as mocked:
        with pytest.raises(ValueError):
            validate_data(df, ["b"])

    assert not mocked.called


def test_validate_data_propagates_schema_error() -> None:
    df = pd.DataFrame({"a": [1]})

    with patch.object(
        validation_module.training_schema,
        "validate",
        side_effect=ValueError("schema failed"),
    ):
        with pytest.raises(ValueError, match="schema failed"):
            validate_data(df, ["a"])


def test_validate_data_does_not_raise_on_valid_data() -> None:
    df = pd.DataFrame({"a": [1]})

    with patch.object(validation_module.training_schema, "validate", MagicMock()):
        validate_data(df, ["a"])
