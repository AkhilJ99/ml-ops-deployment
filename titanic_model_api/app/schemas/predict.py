from typing import Any, List, Optional

from pydantic import BaseModel
from titanic_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Age": 79,
                        "Sex": "M",
                        "BP": "HIGH",
                        "Cholesterol": "NORMAL",
                        "Na_to_K": 11.037,
                    }
                ]
            }
        }
