import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import config as cfg
import pipeline as pl


def activate_training():
    data = pd.read_csv(cfg.DATA_NAME)
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
