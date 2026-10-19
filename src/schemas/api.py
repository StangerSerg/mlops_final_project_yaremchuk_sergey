from pydantic import BaseModel, ConfigDict, Field


class PredictPurchaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    itemid: int = Field(98113, ge=0)
    hour: int = Field(12, ge=0, le=23)
    weekday: int = Field(3, ge=0, le=6)


class PredictPurchaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    itemid: int
    purchase_probability: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    ranker_loaded: bool
    items_in_feature_store: int


class VersionResponse(BaseModel):
    service: str
    version: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
