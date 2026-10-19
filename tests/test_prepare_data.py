from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.ingestion import load_data, load_parquet_data
from src.data.preprocessing import (
    drop_inconsistent_ctr,
    preprocess_events,
    preprocess_properties,
)
from src.features.feature_engineering import (
    build_target,
    compute_item_popularity,
    prepare_features,
    report_ctr_anomalies,
)


def test_load_data_returns_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert isinstance(result, pd.DataFrame)


def test_load_data_is_not_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert not result.empty


def test_load_data_preserves_row_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert len(result) == 3


def test_load_data_preserves_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert list(result.columns) == ["a", "b"]


def test_load_parquet_data_returns_dataframe(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(pq_path, index=False)

    result = load_parquet_data(pq_path)

    assert isinstance(result, pd.DataFrame)


def test_load_parquet_data_is_not_empty(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(pq_path, index=False)

    result = load_parquet_data(pq_path)

    assert not result.empty


def test_load_parquet_data_preserves_row_count(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq_path, index=False)

    result = load_parquet_data(pq_path)

    assert len(result) == 3


def test_load_parquet_data_preserves_columns(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(pq_path, index=False)

    result = load_parquet_data(pq_path)

    assert list(result.columns) == ["a", "b"]


def test_preprocess_events_lowercases_columns() -> None:
    df = pd.DataFrame(
        {
            "ItemID": [1],
            "Property": ["x"],
            "Value": [1],
            "Event": ["view"],
            "Timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert "itemid" in result.columns


def test_preprocess_events_strips_column_whitespace() -> None:
    df = pd.DataFrame(
        {
            " itemid ": [1],
            "property": ["x"],
            "value": [1],
            "event": ["view"],
            "timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert "itemid" in result.columns


def test_preprocess_events_converts_timestamp_to_datetime() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["view"],
            "timestamp": [1_600_000_000_000],
        }
    )

    result = preprocess_events(df)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_preprocess_events_purchase_event_is_true() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["purchase"],
            "timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert bool(result["is_purchase_event"].iloc[0]) is True


def test_preprocess_events_transaction_event_is_true() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["transaction"],
            "timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert bool(result["is_purchase_event"].iloc[0]) is True


def test_preprocess_events_view_event_is_false() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["view"],
            "timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert bool(result["is_purchase_event"].iloc[0]) is False


def test_preprocess_events_event_case_insensitive() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["PURCHASE"],
            "timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert bool(result["is_purchase_event"].iloc[0]) is True


def test_preprocess_events_transactionid_makes_purchase_true() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["view"],
            "timestamp": [0],
            "transactionid": [42],
        }
    )

    result = preprocess_events(df)

    assert bool(result["is_purchase_event"].iloc[0]) is True


def test_preprocess_events_adds_is_purchase_event_column() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["x"],
            "value": [1],
            "event": ["view"],
            "timestamp": [0],
        }
    )

    result = preprocess_events(df)

    assert "is_purchase_event" in result.columns


def _props_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemid": [1, 1, 2],
            "property": ["categoryid", "available", "categoryid"],
            "value": ["10", "1", "20"],
        }
    )


def test_preprocess_properties_lowercases_columns() -> None:
    df = pd.DataFrame(
        {
            "ItemID": [1],
            "Property": ["categoryid"],
            "Value": ["10"],
        }
    )

    result = preprocess_properties(df)

    assert "itemid" in result.columns


def test_preprocess_properties_keeps_only_interaction_columns() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["categoryid"],
            "value": ["10"],
            "extra": ["junk"],
        }
    )

    result = preprocess_properties(df)

    assert set(result.columns) == {"itemid", "categoryid", "available"}


def test_preprocess_properties_filters_unknown_properties() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1, 1],
            "property": ["categoryid", "brand"],
            "value": ["10", "acme"],
        }
    )

    result = preprocess_properties(df)

    assert "brand" not in result.columns


def test_preprocess_properties_returns_one_row_per_item() -> None:
    result = preprocess_properties(_props_df())

    assert len(result) == 2


def test_preprocess_properties_categoryid_is_int() -> None:
    result = preprocess_properties(_props_df())

    assert result["categoryid"].dtype == int


def test_preprocess_properties_available_is_int() -> None:
    result = preprocess_properties(_props_df())

    assert result["available"].dtype == int


def test_preprocess_properties_missing_categoryid_defaults_to_minus_one() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["available"],
            "value": ["1"],
        }
    )

    result = preprocess_properties(df)

    assert result["categoryid"].iloc[0] == -1


def test_preprocess_properties_missing_available_defaults_to_zero() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["categoryid"],
            "value": ["10"],
        }
    )

    result = preprocess_properties(df)

    assert result["available"].iloc[0] == 0


def test_preprocess_properties_invalid_categoryid_becomes_minus_one() -> None:
    df = pd.DataFrame(
        {
            "itemid": [1],
            "property": ["categoryid"],
            "value": ["abc"],
        }
    )

    result = preprocess_properties(df)

    assert result["categoryid"].iloc[0] == -1


def test_preprocess_properties_does_not_mutate_input() -> None:
    df = _props_df()
    original = df.copy()

    preprocess_properties(df)

    pd.testing.assert_frame_equal(df, original)


def _ctr_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemid": [1, 2, 3],
            "ctr": [0.5, 1.5, 0.9],
        }
    )


def test_drop_inconsistent_ctr_removes_high_ctr_rows() -> None:
    result = drop_inconsistent_ctr(_ctr_df(), MagicMock())

    assert 2 not in result["itemid"].tolist()


def test_drop_inconsistent_ctr_keeps_valid_rows() -> None:
    result = drop_inconsistent_ctr(_ctr_df(), MagicMock())

    assert 1 in result["itemid"].tolist()


def test_drop_inconsistent_ctr_boundary_kept() -> None:
    df = pd.DataFrame({"itemid": [1], "ctr": [1.0]})

    result = drop_inconsistent_ctr(df, MagicMock())

    assert len(result) == 1


def test_drop_inconsistent_ctr_resets_index() -> None:
    result = drop_inconsistent_ctr(_ctr_df(), MagicMock())

    assert list(result.index) == [0, 1]


def test_drop_inconsistent_ctr_calls_logger_when_anomalies() -> None:
    logger = MagicMock()

    drop_inconsistent_ctr(_ctr_df(), logger)

    assert logger.info.called


def test_drop_inconsistent_ctr_does_not_log_when_clean() -> None:
    logger = MagicMock()
    df = pd.DataFrame({"itemid": [1], "ctr": [0.5]})

    drop_inconsistent_ctr(df, logger)

    assert not logger.info.called


def test_drop_inconsistent_ctr_no_rows_dropped_keeps_all() -> None:
    df = pd.DataFrame({"itemid": [1, 2], "ctr": [0.1, 0.2]})

    result = drop_inconsistent_ctr(df, MagicMock())

    assert len(result) == 2


def test_drop_inconsistent_ctr_custom_column_name() -> None:
    df = pd.DataFrame({"itemid": [1, 2], "ratio": [0.5, 2.0]})

    result = drop_inconsistent_ctr(df, MagicMock(), ctr_column="ratio")

    assert len(result) == 1


def test_drop_inconsistent_ctr_custom_max() -> None:
    df = pd.DataFrame({"itemid": [1, 2], "ctr": [0.5, 0.7]})

    result = drop_inconsistent_ctr(df, MagicMock(), max_ctr=0.6)

    assert len(result) == 1


def test_drop_inconsistent_ctr_works_without_itemid_column() -> None:
    df = pd.DataFrame({"ctr": [0.5, 1.5]})

    result = drop_inconsistent_ctr(df, MagicMock())

    assert len(result) == 1


# ---------- build_target ----------


def _views() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 10:00:00"]),
            "visitorid": [1],
            "itemid": [100],
        }
    )


def _purchases(ts: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([ts]),
            "visitorid": [1],
            "itemid": [100],
        }
    )


def test_build_target_adds_target_column() -> None:
    result = build_target(_views(), _purchases("2024-01-01 11:00:00"))

    assert "target" in result.columns


def test_build_target_purchase_within_window_is_one() -> None:
    result = build_target(_views(), _purchases("2024-01-01 11:00:00"), window_hours=24)

    assert result["target"].iloc[0] == 1


def test_build_target_purchase_outside_window_is_zero() -> None:
    result = build_target(_views(), _purchases("2024-01-03 10:00:00"), window_hours=24)

    assert result["target"].iloc[0] == 0


def test_build_target_no_purchase_is_zero() -> None:
    empty = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([]),
            "visitorid": [],
            "itemid": [],
        }
    )

    result = build_target(_views(), empty, window_hours=24)

    assert result["target"].iloc[0] == 0


def test_build_target_purchase_before_view_is_zero() -> None:
    result = build_target(_views(), _purchases("2024-01-01 09:00:00"), window_hours=24)

    assert result["target"].iloc[0] == 0


def test_build_target_drops_purchase_time_column() -> None:
    result = build_target(_views(), _purchases("2024-01-01 11:00:00"))

    assert "purchase_time" not in result.columns


def test_build_target_preserves_row_count() -> None:
    result = build_target(_views(), _purchases("2024-01-01 11:00:00"))

    assert len(result) == 1


# ---------- compute_item_popularity ----------


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemid": [1, 1, 1, 2],
            "event": ["view", "view", "purchase", "view"],
            "is_purchase_event": [False, False, True, False],
        }
    )


def test_compute_item_popularity_has_views_column() -> None:
    result = compute_item_popularity(_events())

    assert "views" in result.columns


def test_compute_item_popularity_has_purchases_column() -> None:
    result = compute_item_popularity(_events())

    assert "purchases" in result.columns


def test_compute_item_popularity_has_ctr_column() -> None:
    result = compute_item_popularity(_events())

    assert "ctr" in result.columns


def test_compute_item_popularity_counts_views() -> None:
    result = compute_item_popularity(_events())

    assert result.loc[result["itemid"] == 1, "views"].iloc[0] == 2


def test_compute_item_popularity_counts_purchases() -> None:
    result = compute_item_popularity(_events())

    assert result.loc[result["itemid"] == 1, "purchases"].iloc[0] == 1


def test_compute_item_popularity_no_purchases_zero_filled() -> None:
    result = compute_item_popularity(_events())

    assert result.loc[result["itemid"] == 2, "purchases"].iloc[0] == 0


# ---------- report_ctr_anomalies ----------


def test_report_ctr_anomalies_empty_when_no_anomalies() -> None:
    views = pd.DataFrame({"itemid": [1], "ctr": [0.5]})

    result = report_ctr_anomalies(views)

    assert result.empty


def test_report_ctr_anomalies_returns_rows_with_ctr_above_one() -> None:
    views = pd.DataFrame({"itemid": [1, 2], "ctr": [0.5, 1.5]})

    result = report_ctr_anomalies(views)

    assert len(result) == 1


def test_report_ctr_anomalies_sorted_descending() -> None:
    views = pd.DataFrame({"itemid": [1, 2, 3], "ctr": [2.0, 5.0, 3.0]})

    result = report_ctr_anomalies(views)

    assert result["ctr"].tolist() == [5.0, 3.0, 2.0]


# ---------- prepare_features ----------


def _events_full() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 12:00:00",
                ]
            ),
            "visitorid": [1, 1, 2],
            "itemid": [100, 100, 200],
            "event": ["view", "purchase", "view"],
            "is_purchase_event": [False, True, False],
        }
    )


def _item_props() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemid": [100, 200],
            "categoryid": [10, 20],
            "available": [1, 1],
        }
    )


def test_prepare_features_returns_two_dataframes() -> None:
    item_features, views = prepare_features(_events_full(), _item_props(), 24)

    assert isinstance(item_features, pd.DataFrame)
    assert isinstance(views, pd.DataFrame)


def test_prepare_features_item_features_has_itemid() -> None:
    item_features, _ = prepare_features(_events_full(), _item_props(), 24)

    assert "itemid" in item_features.columns


def test_prepare_features_item_features_has_ctr() -> None:
    item_features, _ = prepare_features(_events_full(), _item_props(), 24)

    assert "ctr" in item_features.columns


def test_prepare_features_views_has_target() -> None:
    _, views = prepare_features(_events_full(), _item_props(), 24)

    assert "target" in views.columns


def test_prepare_features_views_has_hour() -> None:
    _, views = prepare_features(_events_full(), _item_props(), 24)

    assert "hour" in views.columns


def test_prepare_features_views_has_weekday() -> None:
    _, views = prepare_features(_events_full(), _item_props(), 24)

    assert "weekday" in views.columns


def test_prepare_features_views_available_is_int() -> None:
    _, views = prepare_features(_events_full(), _item_props(), 24)

    assert views["available"].dtype == int


def test_prepare_features_views_categoryid_is_int() -> None:
    _, views = prepare_features(_events_full(), _item_props(), 24)

    assert views["categoryid"].dtype == int


def test_prepare_features_unknown_item_gets_categoryid_minus_one() -> None:
    events = _events_full()
    events.loc[2, "itemid"] = 999  # нет в item_props
    _, views = prepare_features(events, _item_props(), 24)

    row = views.loc[views["itemid"] == 999]
    assert row["categoryid"].iloc[0] == -1


def test_prepare_features_unknown_item_gets_available_zero() -> None:
    events = _events_full()
    events.loc[2, "itemid"] = 999
    _, views = prepare_features(events, _item_props(), 24)

    row = views.loc[views["itemid"] == 999]
    assert row["available"].iloc[0] == 0


def test_prepare_features_missing_event_column_raises() -> None:
    events = _events_full().drop(columns=["event"])

    with pytest.raises(KeyError):
        prepare_features(events, _item_props(), 24)


def test_prepare_features_missing_is_purchase_event_raises() -> None:
    events = _events_full().drop(columns=["is_purchase_event"])

    with pytest.raises(KeyError):
        prepare_features(events, _item_props(), 24)


def test_prepare_features_missing_itemid_in_props_raises() -> None:
    props = _item_props().drop(columns=["itemid"])

    with pytest.raises(KeyError):
        prepare_features(_events_full(), props, 24)


def test_prepare_features_returns_only_views_in_views_df() -> None:
    _, views = prepare_features(_events_full(), _item_props(), 24)

    # в выборку попадают только 2 просмотра, покупка (visitorid=1, itemid=100)
    # используется лишь для таргета
    assert len(views) == 2
    assert views["target"].tolist() == [1, 0]


def test_compute_item_popularity_counts_are_float_without_missing() -> None:
    # у каждого товара есть и просмотр, и покупка -> NaN после concat нет
    events = pd.DataFrame(
        {
            "itemid": [1, 1],
            "event": ["view", "purchase"],
            "is_purchase_event": [False, True],
        }
    )

    result = compute_item_popularity(events)

    assert result["views"].dtype == float
    assert result["purchases"].dtype == float
