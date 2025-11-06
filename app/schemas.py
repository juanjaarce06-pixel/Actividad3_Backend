from pydantic import BaseModel, Field
from typing import List

class PredictResponse(BaseModel):
    filename: str
    top1_class: str
    top1_prob: float = Field(ge=0, le=1)
    topk: List[str]
    request_id: str

class HealthResponse(BaseModel):
    status: str
    loaded: bool
    classes: List[str]

class ErrorResponse(BaseModel):
    detail: str
