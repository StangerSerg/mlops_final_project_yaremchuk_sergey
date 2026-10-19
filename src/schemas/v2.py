"""Схемы Open Inference Protocol (KServe V2), REST/JSON.

https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/
"""

from typing import Any

from pydantic import BaseModel, Field


class TensorMetadata(BaseModel):
    name: str
    datatype: str
    shape: list[int]


class ServerMetadataResponse(BaseModel):
    name: str
    version: str
    extensions: list[str] = Field(default_factory=list)


class ModelMetadataResponse(BaseModel):
    name: str
    versions: list[str]
    platform: str
    inputs: list[TensorMetadata]
    outputs: list[TensorMetadata]


class LiveResponse(BaseModel):
    live: bool


class ReadyResponse(BaseModel):
    ready: bool


class InferInputTensor(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[Any]
    parameters: dict[str, Any] | None = None


class InferRequest(BaseModel):
    id: str | None = None
    parameters: dict[str, Any] | None = None
    inputs: list[InferInputTensor] = Field(..., min_length=1)


class InferOutputTensor(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[Any]


class InferResponse(BaseModel):
    model_name: str
    model_version: str | None = None
    id: str | None = None
    outputs: list[InferOutputTensor]


class V2ErrorResponse(BaseModel):
    error: str
