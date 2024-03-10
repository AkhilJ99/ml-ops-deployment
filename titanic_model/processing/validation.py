import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from titanic_model.config.core import config
from titanic_model.processing.data_manager import drug_pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = drug_pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed.copy()
    errors = None

    return validated_data, errors


class DataInputSchema(BaseModel):
    Age:Optional[float]
    Sex: Optional[str]
    BP: Optional[str]
    Cholesterol: Optional[str]
    Na_to_K: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
