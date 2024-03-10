import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder, OrdinalEncoder

from titanic_model.config.core import config
from titanic_model.pipeline import drugs_pipe
from titanic_model.processing.data_manager import *
from sklearn.feature_selection import VarianceThreshold 
from sklearn.neighbors import KNeighborsClassifier




def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_drug_dataset(file_name='datasets/drug200.csv')
    print(data.head())

    # split into independent variables and dependent variable
    X = data.drop('Drug', axis=1)
    y = data['Drug']

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,
        test_size=1/3,
        # we are setting the random seed here
        # for reproducibility
        random_state=0,
    )

    # Pipeline fitting
    drugs_pipe.fit(X_train,y_train)
    y_pred = drugs_pipe.predict(X_test)
    print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist=drugs_pipe)



    
if __name__ == "__main__":
    run_training()
