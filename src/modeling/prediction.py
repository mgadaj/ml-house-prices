import pandas as pd
import joblib
import src.config as cfg
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def create_prediction(test_data):
    regressor = joblib.load(filename=cfg.MODEL_NAME)
    prediction = regressor.predict(test_data)
    return prediction


if __name__ == "__main__":
    path_db = Path(__file__).parent.parent.parent / cfg.DATA_NAME
    data = pd.read_csv(path_db)
    X = data[cfg.SELECTED_VARIABLES]
    y = data[cfg.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    pred = create_prediction(X_test)

    r2 = r2_score(y_test, np.exp(pred))
    mse = mean_squared_error(y_test, np.exp(pred))
    rmse = np.sqrt(mse)

    print(r2)
    print(rmse)