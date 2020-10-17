import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

X_train = pd.read_csv('../data/processed/train_processed.csv')
X_test = pd.read_csv('../data/processed/test_processed.csv')

y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

features = pd.read_csv('../notebooks/features/selected_features.csv')
features = features['0'].to_list()

X_train = X_train[features]
X_test = X_test[features]

linear_model = Lasso(alpha=0.005, random_state=0)
linear_model.fit(X_train, y_train)

pred_train = linear_model.predict(X_train)
rmse_train = sqrt(mean_squared_error(np.exp(y_train), np.exp(pred_train)))
r2_train = r2_score(np.exp(y_train), np.exp(pred_train))

print(f'Train RMSE: {rmse_train}')
print(f'Train R2: {r2_train}')

pred_test = linear_model.predict(X_test)
rmse_test = sqrt(mean_squared_error(np.exp(y_test), np.exp(pred_test)))
r2_test = r2_score(np.exp(y_test), np.exp(pred_test))

print(f'Test RMSE: {rmse_test}')
print(f'Test R2: {r2_test}')

