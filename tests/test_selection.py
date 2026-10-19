import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.models.selection import load_search_space, select_best, split_train_valid_test
from src.models.training import train_model
from src.schemas.config import ModelParams, SplitParams


def _xy(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((x["f1"] + rng.normal(scale=0.5, size=n) > 1.2).astype(int))
    return x, y


# ---------- split_train_valid_test ----------


def test_split_sizes_add_up() -> None:
    x, y = _xy()

    x_tr, x_va, x_te, *_ = split_train_valid_test(x, y, SplitParams())

    assert len(x_tr) + len(x_va) + len(x_te) == len(x)


def test_split_parts_do_not_overlap() -> None:
    x, y = _xy()

    x_tr, x_va, x_te, *_ = split_train_valid_test(x, y, SplitParams())

    assert not set(x_tr.index) & set(x_va.index)
    assert not set(x_tr.index) & set(x_te.index)
    assert not set(x_va.index) & set(x_te.index)


def test_split_is_stratified() -> None:
    x, y = _xy()

    *_, y_tr, y_va, y_te = split_train_valid_test(x, y, SplitParams())

    for part in (y_tr, y_va, y_te):
        assert part.mean() == pytest.approx(y.mean(), abs=0.03)


def test_split_is_deterministic() -> None:
    x, y = _xy()

    first = split_train_valid_test(x, y, SplitParams())
    second = split_train_valid_test(x, y, SplitParams())

    assert list(first[2].index) == list(second[2].index)


# ---------- load_search_space ----------


def test_load_search_space_overrides_base(tmp_path: Path) -> None:
    path = tmp_path / "space.json"
    path.write_text(json.dumps([{"depth": 6}, {"learning_rate": 0.05}]))

    candidates = load_search_space(path, ModelParams(iterations=50))

    assert [c.depth for c in candidates] == [6, 4]
    assert [c.learning_rate for c in candidates] == [0.1, 0.05]
    assert all(c.iterations == 50 for c in candidates)


def test_load_search_space_rejects_unknown_param(tmp_path: Path) -> None:
    path = tmp_path / "space.json"
    path.write_text(json.dumps([{"dept": 6}]))

    with pytest.raises(ValidationError):
        load_search_space(path, ModelParams())


def test_load_search_space_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "space.json"
    path.write_text("[]")

    with pytest.raises(ValueError):
        load_search_space(path, ModelParams())


def test_repo_search_space_is_valid() -> None:
    candidates = load_search_space(
        Path("mlops/clearml/search_space.json"), ModelParams()
    )

    assert len(candidates) >= 2


# ---------- select_best ----------


def test_select_best_returns_max_metric_index() -> None:
    results = [{"pr_auc": 0.1}, {"pr_auc": 0.3}, {"pr_auc": 0.2}]

    assert select_best(results) == 1


def test_select_best_custom_metric() -> None:
    results = [{"pr_auc": 0.3, "roc_auc": 0.5}, {"pr_auc": 0.1, "roc_auc": 0.9}]

    assert select_best(results, metric="roc_auc") == 1


def test_select_best_empty_raises() -> None:
    with pytest.raises(ValueError):
        select_best([])


# ---------- train_model ----------


def test_train_model_with_eval_set_uses_early_stopping() -> None:
    x, y = _xy()
    x_tr, x_va, _, y_tr, y_va, _ = split_train_valid_test(x, y, SplitParams())

    model = train_model(
        x_tr, y_tr, ModelParams(iterations=300, learning_rate=0.3), (x_va, y_va)
    )

    assert model.get_best_iteration() is not None
    assert model.tree_count_ <= 300
