import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Impute missing values for categorical variable
class ImputerCategorical(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X_train, y_train=None):
        return self

    def transform(self, X_train):
        X_train = X_train.copy()
        for variable in variables:
            X_train[variable] = X_train[variable].fillna("Missing")
        return X_train


# Impute missing values for categorical variable
class ImputerNumerical(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X_train, y_train=None):
        self.imputer_dict = {}

        for variable in self.variables:
            self.imputer_dict[variable] = X_train[variable].mode()[0]
        return self

    def transform(self, X_train):
        X_train = X_train.copy()
        for variable in self.variables:
            X_train[variable].fillna(self.imputer_dict[variable], inplace=True)
        return X_train

