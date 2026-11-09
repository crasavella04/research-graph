from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from research_graph.ml.prepare import duplicate_key

Label = Literal["no_issue", "low", "medium", "high"]
Text = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)
]


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Text

    @field_validator("text")
    @classmethod
    def require_words(cls, value: str) -> str:
        if not duplicate_key(value):
            raise ValueError("Text must contain words or numbers")
        return value


class FeedbackRequest(TextRequest):
    label: Label


class Prediction(BaseModel):
    criticality: Literal[0, 1, 2] | None
    label: Label
    confidence: float | None = Field(default=None, ge=0, le=1)


class DatasetSummary(BaseModel):
    dataset_version: str
    rows: int
    labels: dict[str, int]
    pending_count: int


class DatasetUploadResult(DatasetSummary):
    status: Literal["uploaded"]


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None
    training: bool
    dataset: DatasetSummary | None = None


class FeedbackResult(BaseModel):
    id: str
    label: Label
    dataset_version: str
    action: Literal["added", "updated"]
    pending_count: int


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    c: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_iter: int = Field(default=1000, ge=1)


class PredictionRecord(BaseModel):
    id: str
    group_id: str
    text: str
    label: Label
    predicted_label: Label
    confidence: float
    correct: bool


class TrainingResult(BaseModel):
    status: Literal["completed"]
    model_version: str
    dataset_version: str
    duration_seconds: float
    split_sizes: dict[str, int]
    metrics: dict[str, Any]
    predictions: dict[str, list[PredictionRecord]]
