import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Impute missing values for categorical variable
class ImputerCategoricalVar(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].fillna("Missing")
        return X


# Impute missing values for categorical variable
class ImputerNumericalVar(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict = {}
        for variable in self.variables:
            self.imputer_dict[variable] = X[variable].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable].fillna(self.imputer_dict[variable], inplace=True)
        return X


class ProcessorTemporalVar(BaseEstimator, TransformerMixin):


class EncoderRareLabel(BaseEstimator, TransformerMixin):


class EncoderCategoricalVar(BaseEstimator, TransformerMixin):


class TransformerLog(BaseEstimator, TransformerMixin):


