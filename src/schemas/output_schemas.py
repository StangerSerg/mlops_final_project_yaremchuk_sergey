from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SplitParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_state: int = 42
    test_size: float = Field(0.2, gt=0.0, lt=1.0)


class ModelParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    depth: int = Field(4, ge=1, le=16)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    loss_function: str = "Logloss"
    verbose: bool | int = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # пути
    events_path: Path
    item_props_path: Path
    out_item_feats_path: Path
    out_train_path: Path
    output_model_path: Path

    # подготовка данных
    window_hours: int = Field(24, gt=0)
    target: str
    features: list[str] = Field(..., min_length=1)
    required_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)

    # вложенные блоки
    split_params: SplitParams = SplitParams()
    model_params: ModelParams = ModelParams()


class PredictPurchaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    itemid: int = Field(98113, ge=0)
    hour: int = Field(12, ge=0, le=23)
    weekday: int = Field(3, ge=0, le=6)


class PredictPurchaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    itemid: int
    purchase_probability: float = Field(..., ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str