from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SplitParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_state: int = 42
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    # доля валидации от оставшейся после test части (для подбора гиперпараметров)
    valid_size: float = Field(0.2, gt=0.0, lt=1.0)


class ModelParams(BaseModel):
    """Гиперпараметры CatBoostClassifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    depth: int = Field(4, ge=1, le=16)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    iterations: int = Field(1000, ge=1)
    l2_leaf_reg: float = Field(3.0, gt=0.0)
    auto_class_weights: Literal["Balanced", "SqrtBalanced"] | None = None
    loss_function: str = "Logloss"
    random_seed: int = 42
    verbose: bool | int = False


class Config(BaseModel):
    """ML-параметры пайплайна (config/config.json).

    Здесь только имена файлов: в каком бакете или каталоге они лежат,
    определяют настройки окружения (src/utils/settings.py).
    """

    model_config = ConfigDict(extra="forbid")

    # имена артефактов
    events_file: str = "events.csv"
    item_props_file: str = "item_properties_part1.csv"
    item_features_file: str = "item_features.parquet"
    train_file: str = "data_for_training.parquet"
    model_file: str = "ranker.pkl"

    # подготовка данных
    window_hours: int = Field(24, gt=0)
    target: str
    features: list[str] = Field(..., min_length=1)
    required_columns: list[str] = Field(default_factory=list)

    # вложенные блоки
    split_params: SplitParams = SplitParams()
    model_params: ModelParams = ModelParams()
