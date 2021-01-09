import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import src.config as cfg
import src.data_processing.pipeline as pl
from pathlib import Path


def activate_training():
    path_db = Path(__file__).parent.parent.parent / cfg.DATA_NAME
    data = pd.read_csv(path_db)
    X = data[cfg.SELECTED_VARIABLES]
    y = data[cfg.TARGET]

    # split data into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # logarithm transformation of the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # training the model
    pl.model_pipeline.fit(X_train[cfg.SELECTED_VARIABLES], y_train)
    joblib.dump(pl.model_pipeline, cfg.MODEL_NAME)


if __name__ == "__main__":
    activate_training()
