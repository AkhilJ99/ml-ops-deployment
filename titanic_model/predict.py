import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from titanic_model import __version__ as _version
from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_pipeline
from titanic_model.processing.data_manager import pre_pipeline_preparation
from titanic_model.processing.validation import validate_inputs


drug_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
drugs_pipe= load_pipeline(file_name=drug_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    print("validating data")
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    #validated_data=validated_data.reindex(columns=config.model_config.features)
    print("validated data", validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = drugs_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = drugs_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        # print(results)

    return results

if __name__ == "__main__":

    data_in={'Age':[23],'Sex':['F'],'BP':["HIGH"],'Cholesterol':['HIGH'],'Na_to_K':[25.355]}
    
    make_prediction(input_data=data_in)
